import numpy as np
import torch

from algorithms.gnn_mappo.actor import GNNActor
from algorithms.gnn_mappo.critic import GNNCritic
from algorithms.gnn_mappo.graph_builder import build_dynamic_graph
from algorithms.gnn_mappo.graph_encoder import GraphEncoder
from algorithms.hierarchical_marl.high_level_policy import HighLevelPolicy
from algorithms.hierarchical_marl.low_level_policy import LowLevelPolicy
from envs.counter_uav_env import CounterUAVEnv, make_default_config
from envs.scenarios import apply_scenario_to_config


def test_dynamic_graph_contains_expected_nodes_and_edges() -> None:
    env = CounterUAVEnv(apply_scenario_to_config(make_default_config(), "ScenarioA"))
    _, info = env.reset(seed=1)
    graph = build_dynamic_graph(info[env.defense_agents[0]], env.protected_asset, env.config.world_size)
    assert graph.node_features.shape[0] == env.config.num_defenders + env.config.num_intruders + 1
    assert graph.node_features.shape[1] == 10
    assert graph.edge_features.shape[1] == 5
    assert graph.adjacency.shape[0] == graph.node_features.shape[0]


def test_graph_encoder_actor_and_critic_shapes() -> None:
    encoder = GraphEncoder(node_dim=10, hidden_dim=16)
    node_features = torch.zeros((5, 10))
    adjacency = torch.eye(5)
    encoded = encoder(node_features, adjacency)
    actor = GNNActor(obs_dim=8, graph_embedding_dim=16, action_dim=2, hidden_dim=16)
    critic = GNNCritic(state_dim=12, graph_embedding_dim=16, hidden_dim=16)
    actions, log_probs, entropy = actor.sample(torch.zeros((2, 8)), encoded[:2])
    values = critic(torch.zeros((2, 12)), encoder.pool(encoded).repeat(2, 1))
    assert actions.shape == (2, 2)
    assert log_probs.shape == (2,)
    assert entropy.shape == (2,)
    assert values.shape == (2,)


def test_hierarchical_policy_outputs_bounded_actions() -> None:
    high = HighLevelPolicy(high_level_interval=2)
    low = LowLevelPolicy()
    info = {
        "threat_scores": np.array([0.9, 0.1], dtype=np.float32),
        "intruder_positions": np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float32),
        "protected_asset_position": np.array([0.0, 0.0], dtype=np.float32),
    }
    high_action = high.select_action(0, info)
    actions = low.act(
        high_action,
        defender_positions=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        defender_velocities=np.zeros((2, 2), dtype=np.float32),
        intruder_positions=info["intruder_positions"],
        protected_asset_position=info["protected_asset_position"],
    )
    assert actions.shape == (2, 2)
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)
