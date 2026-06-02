from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from algorithms.gnn_mappo.trainer import GNNMAPPOTrainer
from envs.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_gnn_mappo.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    GNNMAPPOTrainer(total_steps=int(config["training"]["total_steps"])).train()


if __name__ == "__main__":
    main()
