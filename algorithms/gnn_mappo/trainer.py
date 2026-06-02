from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from algorithms.gnn_mappo.actor import GNNActor
from algorithms.gnn_mappo.critic import GNNCritic
from algorithms.gnn_mappo.graph_builder import build_dynamic_graph
from algorithms.gnn_mappo.graph_encoder import GraphEncoder
from algorithms.mappo.utils import RunningMeanStd, clip_gradients, normalize_advantages, set_seed
from envs.counter_uav_env import CounterUAVEnv


@dataclass
class GNNMAPPOConfig:
    total_steps: int = 100_000
    rollout_length: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 512
    epochs: int = 4
    max_grad_norm: float = 0.5
    seed: int = 42
    log_dir: str = "experiments/results/gnn_mappo"
    checkpoint_dir: str = "experiments/results/gnn_mappo/checkpoints"
    message_passing_steps: int = 2


class GNNMAPPOTrainer:
    def __init__(self, env: CounterUAVEnv, config: GNNMAPPOConfig, hidden_dim: int = 128):
        set_seed(config.seed)
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agents = env.defense_agents
        self.num_agents = len(self.agents)
        self.obs, self.info = env.reset(seed=config.seed)
        graph = build_dynamic_graph(self.info[self.agents[0]], env.protected_asset, env.config.world_size)
        self.obs_dim = next(iter(self.obs.values())).shape[0]
        self.state_dim = self.info[self.agents[0]]["global_state"].shape[0]
        self.node_dim = graph.node_features.shape[1]
        self.encoder = GraphEncoder(self.node_dim, hidden_dim, config.message_passing_steps).to(self.device)
        self.actor = GNNActor(self.obs_dim, hidden_dim, 2, hidden_dim).to(self.device)
        self.critic = GNNCritic(self.state_dim, hidden_dim, hidden_dim).to(self.device)
        self.optimizer = Adam(list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.learning_rate)
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.state_rms = RunningMeanStd((self.state_dim,))
        self.writer = SummaryWriter(config.log_dir)
        self.global_step = 0

    def train(self) -> None:
        updates = max(1, self.config.total_steps // (self.config.rollout_length * self.num_agents))
        for _ in range(updates):
            rollout = self.collect_rollouts()
            metrics = self.update(rollout)
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, self.global_step)
        self.save_checkpoint(Path(self.config.checkpoint_dir) / "latest.pt")

    def collect_rollouts(self) -> dict[str, np.ndarray]:
        data: dict[str, list[np.ndarray]] = {key: [] for key in ["obs", "agent_emb", "state", "global_emb", "actions", "log_probs", "rewards", "dones", "values"]}
        for _ in range(self.config.rollout_length):
            obs_arr, state_arr, agent_emb, global_emb = self._current_tensors(update_norm=True)
            with torch.no_grad():
                actions_t, log_probs_t, _ = self.actor.sample(obs_arr, agent_emb)
                values_t = self.critic(state_arr, global_emb)
            actions = actions_t.cpu().numpy()
            next_obs, rewards, terms, truncs, next_info = self.env.step(actions)
            reward_arr = np.asarray([rewards[agent] for agent in self.agents], dtype=np.float32)
            done_arr = np.full(self.num_agents, float(terms["__all__"] or truncs["__all__"]), dtype=np.float32)
            data["obs"].append(obs_arr.detach().cpu().numpy())
            data["agent_emb"].append(agent_emb.detach().cpu().numpy())
            data["state"].append(state_arr.detach().cpu().numpy())
            data["global_emb"].append(global_emb.detach().cpu().numpy())
            data["actions"].append(actions)
            data["log_probs"].append(log_probs_t.detach().cpu().numpy())
            data["rewards"].append(reward_arr)
            data["dones"].append(done_arr)
            data["values"].append(values_t.detach().cpu().numpy())
            self.obs, self.info = next_obs, next_info
            self.global_step += self.num_agents
            if terms["__all__"] or truncs["__all__"]:
                self.obs, self.info = self.env.reset(seed=self.config.seed + self.global_step)
        return {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}

    def update(self, rollout: dict[str, np.ndarray]) -> dict[str, float]:
        advantages, returns = self._gae(rollout["rewards"], rollout["dones"], rollout["values"])
        flat = {key: value.reshape(-1, value.shape[-1]) if value.ndim == 3 else value.reshape(-1) for key, value in rollout.items()}
        flat_adv = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        total = len(flat_adv)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        for _ in range(self.config.epochs):
            indices = np.random.permutation(total)
            for start in range(0, total, self.config.batch_size):
                idx = indices[start : start + self.config.batch_size]
                obs = self._tensor(flat["obs"][idx])
                emb = self._tensor(flat["agent_emb"][idx])
                state = self._tensor(flat["state"][idx])
                global_emb = self._tensor(flat["global_emb"][idx])
                actions = self._tensor(flat["actions"][idx])
                old_log_probs = self._tensor(flat["log_probs"][idx])
                adv = normalize_advantages(self._tensor(flat_adv[idx]))
                ret = self._tensor(flat_returns[idx])
                old_values = self._tensor(flat["values"][idx])
                new_log_probs, entropy = self.actor.evaluate_actions(obs, emb, actions)
                values = self.critic(state, global_emb)
                ratio = torch.exp(new_log_probs - old_log_probs)
                policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv).mean()
                value_loss = 0.5 * torch.max((values - ret).pow(2), (old_values + torch.clamp(values - old_values, -0.2, 0.2) - ret).pow(2)).mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                clip_gradients(torch.nn.ModuleList([self.encoder, self.actor, self.critic]), self.config.max_grad_norm)
                self.optimizer.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.mean().item()))
        return {"policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(value_losses)), "entropy": float(np.mean(entropies))}

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder": self.encoder.state_dict(), "actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def _current_tensors(self, update_norm: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = np.stack([self.obs[agent] for agent in self.agents]).astype(np.float32)
        state = self.info[self.agents[0]]["global_state"].astype(np.float32)
        states = np.repeat(state[None, :], self.num_agents, axis=0)
        if update_norm:
            self.obs_rms.update(obs)
            self.state_rms.update(states)
        obs_t = self._tensor(self.obs_rms.normalize(obs))
        state_t = self._tensor(self.state_rms.normalize(states))
        graph = build_dynamic_graph(self.info[self.agents[0]], self.env.protected_asset, self.env.config.world_size)
        node_features = self._tensor(graph.node_features)
        adjacency = self._tensor(graph.adjacency)
        encoded = self.encoder(node_features, adjacency)
        agent_emb = encoded[torch.as_tensor(graph.defender_node_indices, device=self.device)]
        global_emb = self.encoder.pool(encoded).repeat(self.num_agents, 1)
        return obs_t, state_t, agent_emb, global_emb

    def _gae(self, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(self.num_agents, dtype=np.float32)
        for step in reversed(range(len(rewards))):
            next_values = values[step + 1] if step + 1 < len(rewards) else np.zeros(self.num_agents, dtype=np.float32)
            next_nonterminal = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * next_values * next_nonterminal - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * gae
            adv[step] = gae
        return adv, adv + values

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)
