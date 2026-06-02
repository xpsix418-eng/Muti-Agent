from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401
from algorithms.rule_based import NearestIntruderPolicy
from envs.config import load_env_config
from envs.counter_uav_env import CounterUAVEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_2d.yaml")
    args = parser.parse_args()
    env = CounterUAVEnv(load_env_config(args.config))
    _, info = env.reset(seed=42)
    policy = NearestIntruderPolicy(env.config.defender_speed)
    defender_trace = [info["defenders"]]
    intruder_trace = [info["intruders"]]
    for _ in range(env.config.max_steps):
        action = policy.act(info["defenders"], info["intruders"])
        _, _, terminated, truncated, info = env.step(action)
        defender_trace.append(info["defenders"])
        intruder_trace.append(info["intruders"])
        if terminated or truncated:
            break
    plt.figure(figsize=(6, 6))
    for positions in defender_trace:
        plt.scatter(positions[:, 0], positions[:, 1], c="tab:blue", s=8)
    for positions in intruder_trace:
        plt.scatter(positions[:, 0], positions[:, 1], c="tab:red", s=8)
    plt.axis("equal")
    plt.title("Simulated rollout")
    plt.show()


if __name__ == "__main__":
    main()
