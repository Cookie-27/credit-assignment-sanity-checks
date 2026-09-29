# Two-step normalization, smoothing, and update coefficients

This note describes the finite experiment implemented in this repository.
The purpose is a checkable diagnostic, without a novelty or general-performance
claim. The scoring operations are motivated by
[HCAPO §4 and Appendix A](https://arxiv.org/html/2603.08754v1).

## Experiment and objective

Let `p = sigmoid(theta)` be the probability of first action 0. Its success
probability is `q0`; action 1 has success probability `q1`. The second action
is a unique submit operation with probability 1. The terminal reward is binary.

```text
J(theta) = p q0 + (1-p) q1
dJ/dtheta = p(1-p)(q0-q1)
```

The four action/reward outcomes have probabilities
`p(1-q0), p q0, (1-p)(1-q1), (1-p)q1`. A group consists of independent draws
from this distribution. Multinomial count enumeration is exact for the
permutation-invariant group estimator; no Monte Carlo seed is needed.

We report the expected gradient at the first update, with detached advantage
values. A PPO ratio would be 1 at the behavior policy, clipping inactive, and
a KL penalty referenced to that same policy would initially have zero gradient.
Later PPO updates, learned critics, and neural-network optimization are outside
the experiment.

## Scores

The oracle posterior is `h(a|R) = pi(a) P(R|a) / P(R)`. A separate control
reveals the sampled action as part of the outcome, making its posterior 1.
The submit action's posterior is always 1.

Each action is treated as one token. Transform its posterior to `s = h^(1/5)`,
divide the two scores by their within-trajectory mean, and clip each ratio to
`[.8, 1.2]`. The root and terminal weights then satisfy `rho0 + rho1 = 2`.
Root Q is `gamma rho0 R` and terminal Q is `rho1 R`.

For the adjacent blend with root weight `alpha`:

```text
Q_smooth / R = alpha gamma rho0 + (1-alpha) rho1
             = 2(1-alpha) + [alpha(1+gamma)-1] rho0
```

The division notation applies to successful trajectories (`R=1`); both Q
values are zero when `R=0`. With `alpha=.5` and `gamma=.95`, the successful
root value is `1 - .025 rho0`. Thus its order is opposite to the raw root
weight. With `gamma=1, alpha=.5`, it is constant. The coefficient changes sign
at `alpha=1/(1+gamma)`.

Within each rollout group, normalize terminal reward for the macro signal
and root Q for the micro signal using population standard deviation plus
`1e-8`. The protected variants set negative micro values on successful
trajectories to zero. They retain the macro signal. Raw and smoothed variants,
with and without protection, are all reported.

## Why a ranking change is insufficient

For a successful pair with Q difference `d`, positive micro weights sum to
`abs(d)/(abs(d)+2 epsilon)`. Swapping the winner preserves this sum. If both
samples have identical policy inputs and target tokens, they have identical
log-probability gradients; their weighted sum is unchanged by that swap.
For different decisions, equal weight sums do not imply equal gradients.
`coefficient-demo` isolates this algebraic point.

Likewise, when `q0=q1`, changing the policy does not change the terminal-reward
objective. Policy drift should not be reported as lower task success.

## HCA controls and support

[HCA Theorems 1–2](https://proceedings.neurips.cc/paper/9413-hindsight-credit-assignment.pdf)
distinguish marginal future outcomes from outcomes conditioned on the sampled
action. The inverse-ratio return expression needs a nonzero posterior over
the returns in its support. A positive-ratio expression instead integrates
over a marginal future and sums over actions.

The repository includes both reward-conditioned controls and reports their
support separately from the echo experiment. For binary nonnegative rewards,
zero-return terms contribute nothing; overlap on positive-return outcomes is
enough for the implemented expected-gradient identity. The stronger
full-return support flag is also reported. When one action cannot succeed
but another can, the inverse-ratio control is `null`, not a falsely asserted
unbiased gradient. The marginal-future control still evaluates normally.

These identities are existing results, not contributions of this repository.
Outcome representations and spurious credit are also studied in
[COCOA](https://arxiv.org/abs/2306.16803). Our complete-action-echo control does
not support a claim that echo necessarily worsens this particular proxy.

## Interpreting the grid

The full preset contains 240 conditions. Half have complete action echo.
Within each half there are 75 unequal-value and 45 equal-value conditions.
Thresholds for the summary are `true_gradient * proxy_gradient < -1e-12`
and `abs(proxy_gradient) > 1e-10`, respectively. Continuous values remain in
the result file.

This is a diagnostic grid, not a representative sample of agent tasks. The
release contains no LLM calls, native environment rollouts, or training curves.
Expected first-update reversal does not establish worse long-run training.
Ordinary baselines, learning-rate controls, and actual downstream task
performance remain necessary for a learning claim.
