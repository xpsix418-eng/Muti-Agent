from envs.counter_uav_env import make_default_config
from envs.scenarios import SCENARIOS, apply_scenario_to_config, scenario_metadata


def test_scenario_agent_counts() -> None:
    base = make_default_config()
    expected = {
        "ScenarioA": (4, 8),
        "ScenarioB": (8, 16),
        "ScenarioC": (16, 32),
    }
    for name, counts in expected.items():
        config = apply_scenario_to_config(base, name)
        assert (config.num_defenders, config.num_intruders) == counts


def test_all_scenarios_have_metadata() -> None:
    for name in SCENARIOS:
        metadata = scenario_metadata(name)
        assert metadata["name"] == name
        assert "packet_loss" in metadata
