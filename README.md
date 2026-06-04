# counter_uav_marl

`counter_uav_marl` is a Python 3.10+ research project for multi-agent reinforcement learning in simulated 2D/3D cooperative interception-control environments.

This repository is limited to numerical simulation. It does not connect to real UAVs, flight controllers, radio links, weapon systems, targeting systems, or hardware control interfaces.

## Scope

- 5v5 simulated multi-agent cooperative control environment.
- MAPPO, Dense-MAPPO, PI-MAPPO, SA-PMAPPO, Vanilla GNN-MAPPO, and IPG-MAPPO final comparison workflows.
- IPG-MAPPO is the current final candidate method.
- `IPG-MAPPO + Assignment Gate` is kept only as a negative ablation, not as the main method.
- Vanilla GNN-MAPPO is the ordinary graph-network baseline; it does not use interception-point nodes or ITA edge features.

## Install

```bash
conda env create -f environment.yml
conda activate counter_uav_marl
pip install -r requirements-torch-cu128.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Final 5v5 Smoke Test

```bash
python scripts/run_final_5v5_experiments.py \
  --methods vanilla_gnn_mappo ipg_mappo \
  --seeds 1 \
  --total_env_steps 10000 \
  --eval_episodes 5 \
  --results_dir experiments/results/smoke_test
```

## Final 5v5 Experiments

```bash
python scripts/run_final_5v5_experiments.py \
  --methods mappo dense_mappo pi_mappo sa_pmappo vanilla_gnn_mappo ipg_mappo ipg_no_ita ipg_no_graph ipg_with_assignment_gate \
  --seeds 1 2 3 4 5 \
  --total_env_steps 5000000 \
  --eval_episodes 100 \
  --results_dir experiments/results/final_5v5
```

Summarize:

```bash
python scripts/summarize_final_5v5_results.py --results_dir experiments/results/final_5v5
```

Plot curves:

```bash
python scripts/plot_final_training_curves.py --results_dir experiments/results/final_5v5
```

Visualize:

```bash
python scripts/visualize_final_rollouts.py \
  --methods mappo sa_pmappo vanilla_gnn_mappo ipg_mappo ipg_with_assignment_gate \
  --seed 1 \
  --results_dir experiments/results/final_5v5
```

## Direct Training

```bash
python scripts/train_mappo.py --config configs/final_5v5_mappo.yaml
python scripts/train_ipga_mappo.py --config configs/final_5v5_ipg_mappo.yaml
```

## Metrics

- `intercept_rate = intercepted_intruders / total_intruders`; each intruder is counted once per episode.
- `success_rate = 1` only if all intruders are intercepted and no breach occurs in the episode.
- `collision_rate` is kept as the per-step collision rate for backward compatibility.
- `average_collisions_per_step`, `average_collisions_per_episode`, and `collision_episode_rate` are reported separately.

## Safety Boundary

This repository is for algorithmic research inside isolated 2D/3D simulation only. It must not be extended with:

- Real aircraft, UAV, autopilot, simulator bridge, or flight-control APIs.
- Real telemetry, command-and-control, radio, mesh, or datalink interfaces.
- Weapon-system, targeting, interceptor, or payload-control integrations.
- Code intended to guide real-world harm or physical engagement.

All examples use abstract point-mass agents, synthetic goals, synthetic threats, and purely simulated state transitions.

## Layout

- `envs/`: simulated environment, dynamics, threat model, scenarios, rewards, and config parsing.
- `algorithms/mappo/`: MAPPO backbone.
- `algorithms/ipga_mappo/`: IPG-MAPPO, Vanilla GNN-MAPPO switches, and assignment-gate ablation.
- `configs/`: final 5v5 experiment YAML files.
- `scripts/`: final training, evaluation, summarization, curve plotting, and rollout visualization.
- `tests/`: core regression tests for environment, rewards, MAPPO, IPG-MAPPO, and metrics.
