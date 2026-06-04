from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.evaluate import summarize_metrics
from envs.counter_uav_env import CounterUAVEnv, make_default_config


def test_success_and_intercept_rates_are_episode_level() -> None:
    env = CounterUAVEnv(make_default_config())
    rows = [
        {
            "intercept_rate": 1.0,
            "breach_rate": 0.0,
            "high_threat_intercept_rate": 1.0,
            "average_intercept_time": 10.0,
            "average_energy_cost": 1.0,
            "collision_rate": 0.0,
            "communication_cost": 0.0,
            "success_rate": 1.0,
            "average_defender_distance_to_asset": 1.0,
            "average_intercept_distance_to_asset": 100.0,
            "average_intercept_time_to_asset": 10.0,
            "blocking_success_rate": 1.0,
            "assignment_entropy": 0.0,
            "mean_interception_time_advantage": 0.0,
            "graph_attention_sparsity": 0.0,
            "steps": 20.0,
        },
        {
            "intercept_rate": 0.8,
            "breach_rate": 0.2,
            "high_threat_intercept_rate": 0.0,
            "average_intercept_time": 30.0,
            "average_energy_cost": 1.0,
            "collision_rate": 0.0,
            "communication_cost": 0.0,
            "success_rate": 0.0,
            "average_defender_distance_to_asset": 1.0,
            "average_intercept_distance_to_asset": 100.0,
            "average_intercept_time_to_asset": 10.0,
            "blocking_success_rate": 0.0,
            "assignment_entropy": 0.0,
            "mean_interception_time_advantage": 0.0,
            "graph_attention_sparsity": 0.0,
            "steps": 20.0,
        },
    ]
    metrics = summarize_metrics(rows, env, {})
    assert metrics["intercept_rate"] == 0.9
    assert metrics["success_rate"] == 0.5
