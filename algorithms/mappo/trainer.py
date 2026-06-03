from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from algorithms.mappo.actor import MLPActor
from algorithms.mappo.buffer import RolloutBuffer
from algorithms.mappo.critic import MLPCritic
from algorithms.mappo.utils import RunningMeanStd, clip_gradients, linear_lr_schedule, normalize_advantages, set_seed
from envs.counter_uav_env import CounterUAVEnv


@dataclass
class MAPPOConfig:
    total_steps: int = 100_000
    rollout_length: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    batch_size: int = 1024
    epochs: int = 4
    max_grad_norm: float = 0.5
    value_clip: float = 0.2
    reward_normalization: bool = True
    observation_normalization: bool = True
    lr_schedule: bool = True
    seed: int = 42
    log_dir: str = "experiments/results/mappo"
    checkpoint_dir: str = "experiments/results/mappo/checkpoints"


class MAPPOTrainer:
    def __init__(
        self,
        env: CounterUAVEnv,
        config: MAPPOConfig,
        hidden_dim: int = 128,
        device: str | torch.device | None = None,
    ):
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

        self.actor = MLPActor(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic = MLPCritic(self.state_dim, hidden_dim).to(self.device)
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
        )
        self.buffer = RolloutBuffer(
            config.rollout_length,
            self.num_agents,
            self.obs_dim,
            self.state_dim,
            self.action_dim,
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
        self.last_rollout_metrics: dict[str, float] = {}

    def collect_rollouts(self) -> dict[str, float]:
        self.buffer.reset()
        episode_rewards: list[float] = []
        collision_counts: list[float] = []
        energy_costs: list[float] = []
        intercept_rates: list[float] = []
        breach_rates: list[float] = []
        blocking_rates: list[float] = []
        current_episode_reward = 0.0
        current_episode_energy = 0.0
        current_episode_collisions = 0.0
        current_episode_blocking = 0.0
        current_episode_blocking_denominator = 0.0
        current_episode_steps = 0

        for _ in range(self.config.rollout_length):
            obs_array, state_array = self._arrays_from_current_state()
            norm_obs, norm_states = self._normalize_inputs(obs_array, state_array, update=True)
            obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device)
            state_tensor = torch.as_tensor(norm_states, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action_tensor, log_prob_tensor, _ = self.actor.sample(obs_tensor)
                value_tensor = self.critic(state_tensor)
            actions = action_tensor.cpu().numpy()
            log_probs = log_prob_tensor.cpu().numpy()
            values = value_tensor.cpu().numpy()

            if hasattr(self.env, "set_training_step"):
                self.env.set_training_step(self.global_step)
            next_obs, rewards_dict, terminations, truncations, next_info = self.env.step(actions)
            rewards = np.asarray([rewards_dict[agent] for agent in self.agents], dtype=np.float32)
            if self.config.reward_normalization:
                self.reward_rms.update(rewards[None, :])
                rewards = self.reward_rms.normalize(rewards)
            done = float(terminations["__all__"] or truncations["__all__"])
            dones = np.full(self.num_agents, done, dtype=np.float32)
            self.buffer.add(norm_obs, norm_states, actions, log_probs, rewards, dones, values)

            raw_episode_reward = float(np.mean([rewards_dict[agent] for agent in self.agents]))
            current_episode_reward += raw_episode_reward
            current_episode_energy += float(np.mean(np.linalg.norm(actions, axis=1)))
            current_episode_collisions += len(next_info[self.agents[0]]["collision_events"])
            current_episode_blocking += float(np.sum(next_info[self.agents[0]].get("blocking_flags", np.zeros(self.num_agents))))
            current_episode_blocking_denominator += float(self.num_agents)
            current_episode_steps += 1
            self.global_step += self.num_agents
            self.current_obs = next_obs
            self.current_info = next_info

            if done:
                metrics = self._terminal_metrics(
                    current_episode_reward,
                    current_episode_energy,
                    current_episode_collisions,
                    current_episode_blocking,
                    current_episode_blocking_denominator,
                    current_episode_steps,
                )
                episode_rewards.append(metrics["episode_reward"])
                energy_costs.append(metrics["average_energy_cost"])
                collision_counts.append(metrics["collision_rate"])
                intercept_rates.append(metrics["intercept_rate"])
                breach_rates.append(metrics["breach_rate"])
                blocking_rates.append(metrics["blocking_success_rate"])
                self._log_episode_metrics(metrics)
                self.current_obs, self.current_info = self.env.reset(seed=self.config.seed + self.episode_count + 1)
                current_episode_reward = 0.0
                current_episode_energy = 0.0
                current_episode_collisions = 0.0
                current_episode_blocking = 0.0
                current_episode_blocking_denominator = 0.0
                current_episode_steps = 0

        last_obs, last_state = self._arrays_from_current_state()
        _, norm_last_state = self._normalize_inputs(last_obs, last_state, update=False)
        with torch.no_grad():
            last_values = self.critic(torch.as_tensor(norm_last_state, dtype=torch.float32, device=self.device)).cpu().numpy()
        last_dones = np.zeros(self.num_agents, dtype=np.float32)
        self.buffer.compute_returns_and_advantages(last_values, last_dones)
        self.last_rollout_metrics = {
            "episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "intercept_rate": float(np.mean(intercept_rates)) if intercept_rates else 0.0,
            "breach_rate": float(np.mean(breach_rates)) if breach_rates else 0.0,
            "collision_rate": float(np.mean(collision_counts)) if collision_counts else 0.0,
            "average_energy_cost": float(np.mean(energy_costs)) if energy_costs else 0.0,
            "blocking_success_rate": float(np.mean(blocking_rates)) if blocking_rates else 0.0,
        }
        return self.last_rollout_metrics

    def update(self) -> dict[str, float]:
        policy_losses = []
        value_losses = []
        entropies = []
        grad_norms = []
        for _ in range(self.config.epochs):
            for batch in self.buffer.mini_batches(self.config.batch_size):
                advantages = normalize_advantages(batch.advantages)
                new_log_probs, entropy = self.actor.evaluate_actions(batch.observations, batch.actions)
                values = self.critic(batch.global_states)
                ratio = torch.exp(new_log_probs - batch.old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_pred_clipped = batch.old_values + torch.clamp(
                    values - batch.old_values,
                    -self.config.value_clip,
                    self.config.value_clip,
                )
                value_loss_unclipped = (values - batch.returns).pow(2)
                value_loss_clipped = (value_pred_clipped - batch.returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = clip_gradients(nn.ModuleList([self.actor, self.critic]), self.config.max_grad_norm)
                self.optimizer.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                grad_norms.append(grad_norm)

        metrics = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
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
                linear_lr_schedule(self.optimizer, self.config.learning_rate, progress_remaining)
            rollout_metrics = self.collect_rollouts()
            update_metrics = self.update()
            self._log_train_metrics({**rollout_metrics, **update_metrics})
        self.save_checkpoint(Path(self.config.checkpoint_dir) / "latest.pt")

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
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
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.obs_rms.load_state_dict(checkpoint["obs_rms"])
        self.state_rms.load_state_dict(checkpoint["state_rms"])
        self.reward_rms.load_state_dict(checkpoint["reward_rms"])
        self.global_step = int(checkpoint["global_step"])
        self.episode_count = int(checkpoint["episode_count"])

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

    def _terminal_metrics(
        self,
        reward: float,
        energy: float,
        collisions: float,
        blocking: float,
        blocking_denominator: float,
        steps: int,
    ) -> dict[str, float]:
        info = self.current_info[self.agents[0]]
        intercepted = np.asarray(info["intercepted"], dtype=bool)
        breached = np.asarray(info["breached"], dtype=bool)
        metrics = {
            "episode_reward": float(reward),
            "intercept_rate": float(np.mean(intercepted)),
            "breach_rate": float(np.mean(breached)),
            "collision_rate": float(collisions / max(steps, 1)),
            "average_energy_cost": float(energy / max(steps, 1)),
            "blocking_success_rate": float(blocking / max(blocking_denominator, 1.0)),
        }
        self.episode_count += 1
        return metrics

    def _log_episode_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"episode/{key}", value, self.episode_count)

    def _log_update_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)

    def _log_train_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"summary/{key}", value, self.global_step)
