# counter_uav_marl

`counter_uav_marl` is a Python 3.10+ research scaffold for multi-agent reinforcement learning in simulated 2D/3D cooperative control settings.

The project is intentionally limited to numerical simulation. It does not connect to real UAVs, flight controllers, radio links, weapon systems, targeting systems, or any hardware control interface.

## Scope

- Multi-agent simulation environment for cooperative control research.
- Baseline rule-based coordination and assignment.
- MAPPO, graph-based MAPPO, and hierarchical MARL research modules.
- Unit tests for environment reset/step, rewards, threat modeling, graph construction, and evaluation metrics.

## Install

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate counter_uav_marl
```

The environment file records the intended CUDA 12.8 PyTorch setup. If network access to the PyTorch CUDA index is unstable, you can first create the Python 3.10 environment and then install the remaining dependencies:

```bash
conda create -n counter_uav_marl python=3.10 pip
conda activate counter_uav_marl
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Install CUDA 12.8 PyTorch separately from the PyTorch CUDA wheel index:

```bash
pip install -r requirements-torch-cu128.txt
```

Avoid installing PyTorch from a normal PyPI mirror if you need GPU support; those wheels may be CPU-only. Verify CUDA with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Run

Rule-based rollout:

```bash
python scripts/train_rule_based.py --config configs/env_2d.yaml --policy rule_based
python scripts/train_rule_based.py --config configs/env_2d.yaml --policy hungarian
```

MAPPO scaffold:

```bash
python scripts/train_mappo.py --config configs/train_mappo.yaml --scenario ScenarioA
python scripts/train_mappo.py --config configs/train_mappo.yaml --scenario ScenarioB
```

A3 intercept-point reward curriculum:

```bash
python scripts/train_mappo.py --config configs/train_mappo_a3_intercept_point.yaml
```

IPG-MAPPO current main method:

```bash
python scripts/train_ipga_mappo.py --config configs/train_ipg_mappo_5v5.yaml
```

`IPG-MAPPO` is the current tentative final method. It keeps the Interception Prediction Graph, interception-point nodes, Interception-Time Advantage edge feature, graph encoder, and MAPPO backbone. `IPG-MAPPO + Assignment Gate` is treated as an ablation unless the assignment gate is repaired and consistently outperforms IPG-MAPPO.

Strict validation:

```bash
python scripts/run_ipg_validation.py
python scripts/run_ipg_multiseed.py
```

Evaluate a policy:

```bash
python scripts/evaluate.py --config configs/env_2d.yaml --policy rule_based --scenario ScenarioB
python scripts/evaluate.py --config configs/env_2d.yaml --policy hungarian --scenario ScenarioE
python scripts/evaluate.py --config configs/train_ipg_mappo_5v5.yaml --policy ipga_mappo --checkpoint experiments/results/ipg_mappo/Scenario5v5/checkpoints/latest.pt --scenario Scenario5v5
```

## Visualization

`scripts/visualize_rollout.py` runs one simulated rollout and saves:

- `trajectory.png`: static trajectory figure.
- `rollout.gif`: animated rollout.
- `ipga_assignment.png`: graph-method interception points and assignment/attention overlay.
- `ipga_rollout.gif`: graph-method enhanced animation.

Rule-based baseline:

```bash
python scripts/visualize_rollout.py --config configs/env_2d.yaml --policy rule_based --scenario ScenarioB
```

Hungarian assignment baseline:

```bash
python scripts/visualize_rollout.py --config configs/env_2d.yaml --policy hungarian --scenario ScenarioB
```

MAPPO checkpoint:

```bash
python scripts/visualize_rollout.py --config configs/env_2d.yaml --policy mappo --checkpoint experiments/results/mappo/ScenarioA/checkpoints/latest.pt --scenario ScenarioA
```

IPG-MAPPO checkpoint:

```bash
python scripts/visualize_rollout.py --config configs/train_ipg_mappo_5v5.yaml --policy ipga_mappo --checkpoint experiments/results/ipg_mappo/Scenario5v5/checkpoints/latest.pt --scenario Scenario5v5
```

The default output directory is `experiments/results/{policy}_{scenario}_rollout/`; pass `--output-dir` to choose another location. For interactive debugging, call `env.render(mode="human")` from Python to open a live matplotlib view.

## Metrics

- `intercept_rate = intercepted_intruders / total_intruders`; each intruder is counted once per episode.
- `success_rate = 1` only if all intruders are intercepted and no breach occurs in the episode.
- `collision_rate` is reported per episode step.

## Safety Boundary

This repository is for algorithmic research inside isolated 2D/3D simulation only. It must not be extended with:

- Real aircraft, UAV, autopilot, simulator bridge, or flight-control APIs.
- Real telemetry, command-and-control, radio, mesh, or datalink interfaces.
- Weapon-system, targeting, interceptor, or payload-control integrations.
- Code intended to guide real-world harm or physical engagement.

All examples use abstract point-mass agents, synthetic goals, synthetic threats, and purely simulated state transitions.

## Layout

- `envs/`: simulated multi-agent environment, dynamics, threat model, scenarios, and rewards.
- `algorithms/`: baseline and learning algorithm scaffolds.
- `configs/`: YAML configuration files.
- `scripts/`: entry points for training, evaluation, and visualization.
- `tests/`: lightweight regression tests.
- `docs/`: method, environment, and experiment notes.

## Environment API

`CounterUAVEnv` exposes a multi-agent dict API. `reset(seed=None)` returns per-defender observations and info dictionaries. `step(actions)` accepts either a `{agent_id: [ax, ay]}` action mapping or a `(num_defenders, 2)` action array and returns `observations, rewards, terminations, truncations, infos`.

Use `get_global_state()` for centralized critic inputs and `get_observation(agent_id)` for decentralized policy inputs.
