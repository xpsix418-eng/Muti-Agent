from __future__ import annotations

import torch
import numpy as np

from algorithms.ipga_mappo.actor import IPGAActor
from algorithms.ipga_mappo.graph_encoder import InterceptionGraphEncoder
from algorithms.ipga_mappo.interception_graph import InterceptionGraphBuilder
from algorithms.ipga_mappo.soft_assignment_gate import SoftAssignmentGate
from envs.counter_uav_env import CounterUAVEnv, make_default_config


def test_interception_graph_builder_shapes() -> None:
    env = CounterUAVEnv(make_default_config())
    _, info = env.reset(seed=7)
    graph = InterceptionGraphBuilder(
        env.config.world_size,
        env.config.defender_max_speed,
        env.config.intruder_max_speed,
        prediction_horizon=5.0,
    ).build(info[env.defense_agents[0]])

    expected_nodes = env.config.num_defenders + env.config.num_intruders + 1 + env.config.num_intruders
    assert graph.node_features.shape == (expected_nodes, 11)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_features.shape[1] == 7
    assert graph.pair_edge_features.shape == (env.config.num_defenders, env.config.num_intruders, 7)


def test_ipga_encoder_gate_and_actor_outputs_are_valid() -> None:
    env = CounterUAVEnv(make_default_config())
    observations, info = env.reset(seed=9)
    graph = InterceptionGraphBuilder(
        env.config.world_size,
        env.config.defender_max_speed,
        env.config.intruder_max_speed,
        prediction_horizon=5.0,
    ).build(info[env.defense_agents[0]])

    encoder = InterceptionGraphEncoder(11, 7, hidden_dim=32, num_layers=1, attention_heads=4)
    gate = SoftAssignmentGate(graph_hidden_dim=32, edge_dim=7, hidden_dim=32)
    actor = IPGAActor(next(iter(observations.values())).shape[0], graph_dim=32, hidden_dim=32)

    node_features = torch.as_tensor(graph.node_features[None], dtype=torch.float32)
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)
    edge_features = torch.as_tensor(graph.edge_features[None], dtype=torch.float32)
    pair_features = torch.as_tensor(graph.pair_edge_features[None], dtype=torch.float32)
    node_embeddings, _, _ = encoder(node_features, edge_index, edge_features)
    defenders = node_embeddings[:, : env.config.num_defenders]
    intruders = node_embeddings[:, env.config.num_defenders : env.config.num_defenders + env.config.num_intruders]
    point_start = env.config.num_defenders + env.config.num_intruders + 1
    points = node_embeddings[:, point_start : point_start + env.config.num_intruders]
    weights, context = gate(defenders, intruders, points, pair_features)

    obs = torch.as_tensor(np.stack([observations[agent] for agent in env.defense_agents]), dtype=torch.float32)
    actions = actor.deterministic(obs, defenders[0], context[0])
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)
