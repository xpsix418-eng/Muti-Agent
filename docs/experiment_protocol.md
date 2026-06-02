# Experiment Protocol

Recommended experiment flow:

1. Validate reset, step, reward, threat, and graph utilities with `pytest`.
2. Run rule-based rollouts as a sanity baseline.
3. Train MAPPO variants with fixed seeds.
4. Compare success rate, episode return, time-to-capture, and protected-zone breach rate.
5. Store generated logs under `experiments/results/`.
