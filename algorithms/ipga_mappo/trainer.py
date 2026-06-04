from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from algorithms.ipga_mappo.actor import IPGAActor
from algorithms.ipga_mappo.critic import IPGACritic
from algorithms.ipga_mappo.graph_encoder import InterceptionGraphEncoder
from algorithms.ipga_mappo.interception_graph import InterceptionGraphBuilder
from algorithms.ipga_mappo.losses import (
    assignment_loss_weight,
    clipped_policy_loss,
    clipped_value_loss,
    total_ipga_loss,
)
from algorithms.ipga_mappo.soft_assignment_gate import SoftAssignmentGate
from algorithms.ipga_mappo.utils import assignment_entropy, graph_attention_sparsity, mean_interception_time_advantage
from algorithms.mappo.buffer import RolloutBuffer
from algorithms.mappo.utils import RunningMeanStd, clip_gradients, linear_lr_schedule, normalize_advantages, set_seed
from envs.counter_uav_env import CounterUAVEnv


@dataclass
class IPGAMAPPOConfig:
    total_steps: int = 2_000_000
    rollout_length: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.008
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    batch_size: int = 2048
    epochs: int = 5
    max_grad_norm: float = 0.5
    value_clip: float = 0.2
    reward_normalization: bool = True
    observation_normalization: bool = True
    lr_schedule: bool = True
    seed: int = 42
    prediction_horizon: float = 5.0
    hidden_dim: int = 128
    graph_hidden_dim: int = 128
    num_graph_layers: int = 2
    attention_heads: int = 4
    assignment_loss_start: float = 0.04
    assignment_loss_end: float = 0.0
    assignment_loss_decay_steps: int = 1_000_000
    use_graph: bool = True
    use_assignment_gate: bool = True
    use_ita_features: bool = True
    use_assignment_loss: bool = True
    log_dir: str = "experiments/results/ipga_mappo"
    checkpoint_dir: str = "experiments/results/ipga_mappo/checkpoints"


@dataclass
class GraphMiniBatch:
    observations: torch.Tensor
    global_states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    pair_edge_features: torch.Tensor
    heuristic_assignments: torch.Tensor


class IPGAGraphRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        rollout_length: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        node_count: int,
        node_dim: int,
        edge_count: int,
        edge_dim: int,
        num_intruders: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        self.node_count = node_count
        self.node_dim = node_dim
        self.edge_count = edge_count
        self.edge_dim = edge_dim
        self.num_intruders = num_intruders
        super().__init__(rollout_length, num_agents, obs_dim, state_dim, action_dim, gamma, gae_lambda, device)

    def reset(self) -> None:
        super().reset()
        self.node_features = np.zeros((self.rollout_length, self.node_count, self.node_dim), dtype=np.float32)
        self.edge_features = np.zeros((self.rollout_length, self.edge_count, self.edge_dim), dtype=np.float32)
        self.pair_edge_features = np.zeros(
            (self.rollout_length, self.num_agents, self.num_intruders, self.edge_dim),
            dtype=np.float32,
        )
        self.heuristic_assignments = np.full((self.rollout_length, self.num_agents), -1, dtype=np.int64)

    def add_graph(self, graph_data) -> None:
        pos = self.position
        self.node_features[pos] = graph_data.node_features
        self.edge_features[pos] = graph_data.edge_features
        self.pair_edge_features[pos] = graph_data.pair_edge_features
        self.heuristic_assignments[pos] = graph_data.heuristic_assignments

    def graph_mini_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[GraphMiniBatch]:
        step_batch_size = max(1, batch_size // self.num_agents)
        indices = np.arange(self.position)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, self.position, step_batch_size):
            steps = indices[start : start + step_batch_size]
            batch_steps = len(steps)
            yield GraphMiniBatch(
                observations=self._tensor(self.observations[steps].reshape(batch_steps * self.num_agents, self.obs_dim)),
                global_states=self._tensor(self.global_states[steps].reshape(batch_steps * self.num_agents, self.state_dim)),
                actions=self._tensor(self.actions[steps].reshape(batch_steps * self.num_agents, self.action_dim)),
                old_log_probs=self._tensor(self.log_probs[steps].reshape(batch_steps * self.num_agents)),
                advantages=self._tensor(self.advantages[steps].reshape(batch_steps * self.num_agents)),
                returns=self._tensor(self.returns[steps].reshape(batch_steps * self.num_agents)),
                old_values=self._tensor(self.values[steps].reshape(batch_steps * self.num_agents)),
                node_features=self._tensor(self.node_features[steps]),
                edge_features=self._tensor(self.edge_features[steps]),
                pair_edge_features=self._tensor(self.pair_edge_features[steps]),
                heuristic_assignments=torch.as_tensor(self.heuristic_assignments[steps], dtype=torch.long, device=self.device),
            )


class IPGAMAPPOTrainer:
    def __init__(self, env: CounterUAVEnv, config: IPGAMAPPOConfig, device: str | torch.device | None = None):
        set_seed(config.seed)
        self.env = env
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.agents = env.defense_agents
        self.num_agents = len(self.agents)
        obs, info = env.reset(seed=config.seed)
        self.current_obs = obs
        self.current_info = info
        self.obs_dim = next(iter(obs.values())).shape[0]
        self.state_dim = info[self.agents[0]]["global_state"].shape[0]
        self.action_dim = 2
        self.graph_builder = InterceptionGraphBuilder(
            env.config.world_size,
            env.config.defender_max_speed,
            env.config.intruder_max_speed,
            config.prediction_horizon,
        )
        sample_graph = self.graph_builder.build(info[self.agents[0]])
        self.edge_index = torch.as_tensor(sample_graph.edge_index, dtype=torch.long, device=self.device)
        graph_hidden = config.graph_hidden_dim
        self.graph_encoder = InterceptionGraphEncoder(
            sample_graph.node_features.shape[-1],
            sample_graph.edge_features.shape[-1],
            graph_hidden,
            config.num_graph_layers,
            config.attention_heads,
        ).to(self.device)
        self.assignment_gate = SoftAssignmentGate(graph_hidden, sample_graph.edge_features.shape[-1], config.hidden_dim).to(self.device)
        self.actor = IPGAActor(self.obs_dim, graph_hidden, self.action_dim, config.hidden_dim).to(self.device)
        self.critic = IPGACritic(self.state_dim, graph_hidden, config.hidden_dim).to(self.device)
        self.optimizer = Adam(
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.graph_encoder.parameters())
            + list(self.assignment_gate.parameters()),
            lr=config.learning_rate,
        )
        self.buffer = IPGAGraphRolloutBuffer(
            config.rollout_length,
            self.num_agents,
            self.obs_dim,
            self.state_dim,
            self.action_dim,
            sample_graph.node_features.shape[0],
            sample_graph.node_features.shape[1],
            sample_graph.edge_features.shape[0],
            sample_graph.edge_features.shape[1],
            env.config.num_intruders,
            config.gamma,
            config.gae_lambda,
            self.device,
        )
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.state_rms = RunningMeanStd((self.state_dim,))
        self.reward_rms = RunningMeanStd((self.num_agents,))
        self.writer = SummaryWriter(config.log_dir)
        self.global_step = 0
        self.episode_count = 0

    def collect_rollouts(self) -> dict[str, float]:
        self.buffer.reset()
        episode_rewards: list[float] = []
        intercept_rates: list[float] = []
        breach_rates: list[float] = []
        collision_rates: list[float] = []
        blocking_rates: list[float] = []
        energy_costs: list[float] = []
        assignment_entropies: list[float] = []
        ita_values: list[float] = []
        attention_sparsities: list[float] = []
        reward_sum = 0.0
        energy_sum = 0.0
        collision_sum = 0.0
        blocking_sum = 0.0
        blocking_denominator = 0.0
        episode_steps = 0
        for _ in range(self.config.rollout_length):
            obs_array, state_array = self._arrays_from_current_state()
            norm_obs, norm_states = self._normalize_inputs(obs_array, state_array, update=True)
            graph = self.graph_builder.build(self.current_info[self.agents[0]])
            graph_tensors = self._graph_tensors(graph)
            with torch.no_grad():
                defender_embeddings, pooled, context, weights, attention = self._graph_forward(*graph_tensors)
                obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device)
                state_tensor = torch.as_tensor(norm_states, dtype=torch.float32, device=self.device)
                actor_defender_embeddings = defender_embeddings[0]
                actor_context = context[0]
                critic_pooled = pooled
                if not self.config.use_graph:
                    actor_defender_embeddings = torch.zeros_like(actor_defender_embeddings)
                    actor_context = torch.zeros_like(actor_context)
                    critic_pooled = torch.zeros_like(critic_pooled)
                elif not self.config.use_assignment_gate:
                    actor_context = torch.zeros_like(actor_context)
                action_tensor, log_prob_tensor, _ = self.actor.sample(obs_tensor, actor_defender_embeddings, actor_context)
                value_tensor = self.critic(state_tensor, critic_pooled.expand(self.num_agents, -1))
            actions = action_tensor.cpu().numpy()
            values = value_tensor.cpu().numpy()
            rewards, done, next_info = self._env_step(actions)
            if self.config.reward_normalization:
                self.reward_rms.update(rewards[None, :])
                rewards = self.reward_rms.normalize(rewards)
            self.buffer.add_graph(graph)
            self.buffer.add(norm_obs, norm_states, actions, log_prob_tensor.cpu().numpy(), rewards, np.full(self.num_agents, done), values)
            reward_sum += float(np.mean(rewards))
            energy_sum += float(np.mean(np.linalg.norm(actions, axis=1)))
            collision_sum += len(next_info[self.agents[0]]["collision_events"])
            blocking_sum += float(np.sum(next_info[self.agents[0]].get("blocking_flags", np.zeros(self.num_agents))))
            blocking_denominator += float(self.num_agents)
            assignment_entropies.append(assignment_entropy(weights.cpu().numpy()))
            ita_values.append(mean_interception_time_advantage(graph.interception_time_advantage))
            attention_sparsities.append(graph_attention_sparsity(attention.cpu().numpy()))
            episode_steps += 1
            if done:
                info = self.current_info[self.agents[0]]
                intercepted = np.asarray(info["intercepted"], dtype=bool)
                breached = np.asarray(info["breached"], dtype=bool)
                metrics = {
                    "episode_reward": float(reward_sum),
                    "intercept_rate": float(np.mean(intercepted)),
                    "breach_rate": float(np.mean(breached)),
                    "collision_rate": float(collision_sum / max(episode_steps, 1)),
                    "blocking_success_rate": float(blocking_sum / max(blocking_denominator, 1.0)),
                    "average_energy_cost": float(energy_sum / max(episode_steps, 1)),
                }
                self._log_episode_metrics(metrics)
                episode_rewards.append(metrics["episode_reward"])
                intercept_rates.append(metrics["intercept_rate"])
                breach_rates.append(metrics["breach_rate"])
                collision_rates.append(metrics["collision_rate"])
                blocking_rates.append(metrics["blocking_success_rate"])
                energy_costs.append(metrics["average_energy_cost"])
                self.current_obs, self.current_info = self.env.reset(seed=self.config.seed + self.episode_count + 1)
                reward_sum = energy_sum = collision_sum = blocking_sum = blocking_denominator = 0.0
                episode_steps = 0
        last_obs, last_state = self._arrays_from_current_state()
        _, norm_last_state = self._normalize_inputs(last_obs, last_state, update=False)
        last_graph = self.graph_builder.build(self.current_info[self.agents[0]])
        with torch.no_grad():
            _, pooled, _, _, _ = self._graph_forward(*self._graph_tensors(last_graph))
            last_values = self.critic(
                torch.as_tensor(norm_last_state, dtype=torch.float32, device=self.device),
                pooled.expand(self.num_agents, -1),
            ).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, np.zeros(self.num_agents, dtype=np.float32))
        return {
            "episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "intercept_rate": float(np.mean(intercept_rates)) if intercept_rates else 0.0,
            "breach_rate": float(np.mean(breach_rates)) if breach_rates else 0.0,
            "collision_rate": float(np.mean(collision_rates)) if collision_rates else 0.0,
            "blocking_success_rate": float(np.mean(blocking_rates)) if blocking_rates else 0.0,
            "average_energy_cost": float(np.mean(energy_costs)) if energy_costs else 0.0,
            "assignment_entropy": float(np.mean(assignment_entropies)) if assignment_entropies else 0.0,
            "mean_interception_time_advantage": float(np.mean(ita_values)) if ita_values else 0.0,
            "graph_attention_sparsity": float(np.mean(attention_sparsities)) if attention_sparsities else 0.0,
        }

    def update(self) -> dict[str, float]:
        policy_losses = []
        value_losses = []
        entropies = []
        assignment_losses = []
        grad_norms = []
        lambda_assign = assignment_loss_weight(
            self.global_step,
            self.config.assignment_loss_start,
            self.config.assignment_loss_end,
            self.config.assignment_loss_decay_steps,
        )
        if not self.config.use_assignment_loss:
            lambda_assign = 0.0
        for _ in range(self.config.epochs):
            for batch in self.buffer.graph_mini_batches(self.config.batch_size):
                advantages = normalize_advantages(batch.advantages)
                defender_embeddings, pooled, context, weights, _ = self._graph_forward(
                    batch.node_features,
                    batch.edge_features,
                    batch.pair_edge_features,
                )
                batch_steps = defender_embeddings.shape[0]
                selected_defenders = defender_embeddings.reshape(batch_steps * self.num_agents, -1)
                selected_context = context.reshape(batch_steps * self.num_agents, -1)
                pooled = pooled[:, None, :].expand(batch_steps, self.num_agents, -1).reshape(batch_steps * self.num_agents, -1)
                if not self.config.use_graph:
                    selected_defenders = torch.zeros_like(selected_defenders)
                    selected_context = torch.zeros_like(selected_context)
                    pooled = torch.zeros_like(pooled)
                if not self.config.use_assignment_gate:
                    selected_context = torch.zeros_like(selected_context)
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch.observations,
                    selected_defenders,
                    selected_context,
                    batch.actions,
                )
                values = self.critic(batch.global_states, pooled)
                policy_loss = clipped_policy_loss(new_log_probs, batch.old_log_probs, advantages, self.config.clip_ratio)
                value_loss = clipped_value_loss(values, batch.old_values, batch.returns, self.config.value_clip)
                assignment_aux_loss = self.assignment_gate.auxiliary_loss(weights, batch.heuristic_assignments)
                loss_output = total_ipga_loss(
                    policy_loss,
                    value_loss,
                    entropy.mean(),
                    assignment_aux_loss,
                    self.config.value_coef,
                    self.config.entropy_coef,
                    lambda_assign,
                )
                self.optimizer.zero_grad()
                loss_output.total_loss.backward()
                grad_norm = clip_gradients(nn.ModuleList([self.actor, self.critic, self.graph_encoder, self.assignment_gate]), self.config.max_grad_norm)
                self.optimizer.step()
                policy_losses.append(float(loss_output.policy_loss.item()))
                value_losses.append(float(loss_output.value_loss.item()))
                entropies.append(float(loss_output.entropy.item()))
                assignment_losses.append(float(loss_output.assignment_loss.item()))
                grad_norms.append(grad_norm)
        metrics = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "assignment_loss": float(np.mean(assignment_losses)),
            "assignment_loss_weight": float(lambda_assign),
            "grad_norm": float(np.mean(grad_norms)),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        self._log_update_metrics(metrics)
        return metrics

    def train(self) -> None:
        updates = max(1, self.config.total_steps // (self.config.rollout_length * self.num_agents))
        for update_idx in range(updates):
            progress_remaining = 1.0 - update_idx / max(updates, 1)
            if self.config.lr_schedule:
                linear_lr_schedule(
                    self.optimizer,
                    self.config.learning_rate,
                    progress_remaining,
                    self.config.min_learning_rate,
                )
            metrics = {**self.collect_rollouts(), **self.update()}
            self._log_train_metrics(metrics)
            if update_idx == 0 or (update_idx + 1) % 10 == 0 or update_idx + 1 == updates:
                print(
                    "update="
                    f"{update_idx + 1}/{updates} "
                    f"step={self.global_step} "
                    f"intercept_rate={metrics.get('intercept_rate', 0.0):.3f} "
                    f"blocking_success_rate={metrics.get('blocking_success_rate', 0.0):.3f} "
                    f"collision_rate={metrics.get('collision_rate', 0.0):.3f} "
                    f"assignment_loss={metrics.get('assignment_loss', 0.0):.4f} "
                    f"lr={metrics.get('learning_rate', 0.0):.6f}",
                    flush=True,
                )
        self.save_checkpoint(Path(self.config.checkpoint_dir) / "latest.pt")

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "graph_encoder": self.graph_encoder.state_dict(),
                "assignment_gate": self.assignment_gate.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_rms": self.obs_rms.state_dict(),
                "state_rms": self.state_rms.state_dict(),
                "reward_rms": self.reward_rms.state_dict(),
                "global_step": self.global_step,
                "episode_count": self.episode_count,
                "config": self.config.__dict__,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.graph_encoder.load_state_dict(checkpoint["graph_encoder"])
        self.assignment_gate.load_state_dict(checkpoint["assignment_gate"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.obs_rms.load_state_dict(checkpoint["obs_rms"])
        self.state_rms.load_state_dict(checkpoint["state_rms"])
        self.reward_rms.load_state_dict(checkpoint["reward_rms"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.episode_count = int(checkpoint.get("episode_count", 0))

    def _env_step(self, actions: np.ndarray) -> tuple[np.ndarray, float, dict]:
        if hasattr(self.env, "set_training_step"):
            self.env.set_training_step(self.global_step)
        next_obs, rewards_dict, terminations, truncations, next_info = self.env.step(actions)
        rewards = np.asarray([rewards_dict[agent] for agent in self.agents], dtype=np.float32)
        done = float(terminations["__all__"] or truncations["__all__"])
        self.current_obs = next_obs
        self.current_info = next_info
        self.global_step += self.num_agents
        return rewards, done, next_info

    def _arrays_from_current_state(self) -> tuple[np.ndarray, np.ndarray]:
        observations = np.stack([self.current_obs[agent] for agent in self.agents]).astype(np.float32)
        global_state = self.current_info[self.agents[0]]["global_state"].astype(np.float32)
        global_states = np.repeat(global_state[None, :], self.num_agents, axis=0)
        return observations, global_states

    def _normalize_inputs(self, obs: np.ndarray, states: np.ndarray, update: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.config.observation_normalization and update:
            self.obs_rms.update(obs)
            self.state_rms.update(states)
        if self.config.observation_normalization:
            return self.obs_rms.normalize(obs), self.state_rms.normalize(states)
        return obs, states

    def _graph_tensors(self, graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = torch.as_tensor(graph.node_features[None, :, :], dtype=torch.float32, device=self.device)
        edge_features = torch.as_tensor(graph.edge_features[None, :, :], dtype=torch.float32, device=self.device)
        pair_features = torch.as_tensor(graph.pair_edge_features[None, :, :, :], dtype=torch.float32, device=self.device)
        return node_features, edge_features, pair_features

    def _graph_forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        pair_edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.config.use_ita_features:
            pair_edge_features = pair_edge_features.clone()
            edge_features = edge_features.clone()
            pair_edge_features[..., 5] = 0.0
            edge_features[..., 5] = 0.0
        node_embeddings, pooled, attention = self.graph_encoder(node_features, self.edge_index, edge_features)
        num_defenders = self.env.config.num_defenders
        num_intruders = self.env.config.num_intruders
        defender_embeddings = node_embeddings[:, :num_defenders]
        intruder_embeddings = node_embeddings[:, num_defenders : num_defenders + num_intruders]
        point_start = num_defenders + num_intruders + 1
        point_embeddings = node_embeddings[:, point_start : point_start + num_intruders]
        weights, context = self.assignment_gate(defender_embeddings, intruder_embeddings, point_embeddings, pair_edge_features)
        return defender_embeddings, pooled, context, weights, attention

    def _log_episode_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"episode/{key}", value, self.episode_count)
        self.episode_count += 1

    def _log_update_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)

    def _log_train_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"summary/{key}", value, self.global_step)
