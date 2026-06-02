from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from envs.config import load_env_config
from envs.counter_uav_env import CounterUAVEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    args = parser.parse_args()
    env = CounterUAVEnv(load_env_config(args.config))
    _, _ = env.reset(seed=42)
    total_reward = 0.0
    for _ in range(env.config.max_steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"evaluation_total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
