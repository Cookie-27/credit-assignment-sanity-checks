# Exact experiment results

Generated with `python -m credit_sanity report --output results/summary.md`.
The command recomputes the experiment; it does not summarize a cached file.

Version: 0.1.0. Conditions: 240.
Eight reward pairs x five policy probabilities x three group sizes x two
outcome representations. No random sampling, model calls, or training.

## Reward-only outcome

| Signal | Opposite directions / unequal-value cases | Drift / equal-value cases |
|---|---:|---:|
| `grpo` | 0 / 75 | 0 / 45 |
| `trajectory_raw` | 12 / 75 | 36 / 45 |
| `trajectory_smooth` | 23 / 75 | 36 / 45 |
| `trajectory_raw_unprotected` | 18 / 75 | 36 / 45 |
| `trajectory_smooth_unprotected` | 24 / 75 | 36 / 45 |

## Complete action echo control

| Signal | Opposite directions / unequal-value cases | Drift / equal-value cases |
|---|---:|---:|
| `grpo` | 0 / 75 | 0 / 45 |
| `trajectory_raw` | 0 / 75 | 0 / 45 |
| `trajectory_smooth` | 0 / 75 | 0 / 45 |
| `trajectory_raw_unprotected` | 0 / 75 | 0 / 45 |
| `trajectory_smooth_unprotected` | 0 / 75 | 0 / 45 |

## Interpretation

- Opposite direction means `true_gradient * proxy_gradient < -1e-12`.
- Equal-value drift means `abs(proxy_gradient) > 1e-10` when
  `abs(true_gradient) <= 1e-12`; the reward is constant in this case.
- These are counts in a fixed diagnostic grid, not failure-rate estimates
  for real agents. They concern the expected first update, not learning curves.
- The complete-echo control removes all counted reversals and tied drift
  in this grid. Preserve this result when discussing the counterexamples.
- Individual values, configurations, probability sums, and HCA support
  flags are recorded in [theory.json](theory.json).

See the [research note](../docs/research_note.md) for the derivation and
[reproducibility guide](../REPRODUCIBILITY.md) for independent checks.
