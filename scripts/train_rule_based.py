from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from algorithms.rule_based import NearestIntruderPolicy
from envs.config import load_env_config
from envs.counter_uav_env import CounterUAVEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    args = parser.parse_args()
    env = CounterUAVEnv(load_env_config(args.config))
    obs, info = env.reset(seed=42)
    del obs
    policy = NearestIntruderPolicy(env.config.defender_speed)
    total_reward = 0.0
    for _ in range(env.config.max_steps):
        action = policy.act(info["defenders"], info["intruders"])
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"rule_based_total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
