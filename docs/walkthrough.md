# Four checks before interpreting a credit score

For RL researchers and students, this walkthrough connects a score to an
expected policy update in a fully specified, finite example. Run commands
from the cloned repository root. Python 3.10+ is the only runtime requirement.

The policy selects action 0 with probability `p`, action 1 otherwise. Each
action succeeds with probability `q0` or `q1`; a deterministic submit step
then ends the episode. The sole trainable parameter is the logit of `p`.
We enumerate every possible rollout-group composition, without sampling.

## 1. Does the proxy point toward more reward?

```bash
python -m credit_sanity demo
```

The true gradient is `+0.000900000000`, while `trajectory_smooth` is
`-0.180982441841`. Here a positive logit update makes action 0 more likely,
so the proxy's negative direction reduces the expected terminal reward.
This is an expected first-update result. It is not a training curve.

You can inspect and change the setting directly in Python:

```pycon
>>> from credit_sanity import Config, evaluate
>>> case = evaluate(Config(q0=1, q1=.99, p=.9, group=8))
>>> format(case["true_gradient"], "+.12f")
'+0.000900000000'
>>> format(case["gradients"]["trajectory_smooth"], "+.12f")
'-0.180982441841'
>>> case["gradients"]["grpo"] > 0
True

```

`grpo` is the group-standardized terminal-reward signal in this one-logit
example, without PPO clipping, KL regularization, or neural-network training.
Both protected trajectory proxies add their micro signal to that macro
signal. See the [derivation](research_note.md) for the exact operations.

## 2. Is policy drift actually harmful?

```bash
python -m credit_sanity demo --q0 1 --q1 1 --p 0.9
```

The smoothed gradient is nonzero, but both actions always succeed. Moving
probability between them cannot change the terminal-reward objective.

```pycon
>>> tied = evaluate(Config(q0=1, q1=1, p=.9))
>>> tied["true_gradient"] == 0
True
>>> abs(tied["gradients"]["trajectory_smooth"]) > .1
True
>>> all(change == 0 for change in tied["finite_step_reward_changes"].values())
True

```

Reporting only the gradient change would miss this distinction.

## 3. Does an outcome representation change the finding?

```bash
python -m credit_sanity demo --echo
```

The complete-echo control includes the sampled action in the outcome. Its
posterior becomes 1; the default example's smoothed gradient is now positive.
All counted reversals and tied drift disappear in the echo half of the fixed
grid. This control limits what the non-echo counterexamples establish.

```pycon
>>> echoed = evaluate(Config(echo=True))
>>> echoed["gradients"]["trajectory_smooth"] > 0
True
>>> all(outcome["posterior"] == 1 for outcome in echoed["outcomes"])
True

```

## 4. Does a changed ranking change the aggregate update?

```bash
python -m credit_sanity coefficient-demo
```

When two samples have the same policy input and target tokens, their
log-probability gradients are identical. Swapping their positive micro-credit
winner preserves the total coefficient in this pair example. Distinct
decisions do not generally share a gradient, so this identity has a narrow scope.

```pycon
>>> from credit_sanity.exact import identical_decision_demo
>>> pair = identical_decision_demo()
>>> pair["before"] != pair["after"]
True
>>> pair["sum_before"] == pair["sum_after"]
True

```

## Explore your own setting

```bash
python -m credit_sanity demo --q0 0.9 --q1 0.5 --p 0.25 --group 4 --alpha 0.7 --json
python -m credit_sanity demo --help
```

Important inputs:

| Input | Meaning | Range |
|---|---|---|
| `q0`, `q1` | Success probability after each first action | [0, 1] |
| `p` | Probability of selecting action 0 | (0, 1) |
| `group` | Independent rollouts per normalized group | Integers 1–32 |
| `alpha` | Root weight in the adjacent blend | [0, 1] |
| `gamma` | Root-credit discount | [0, 1] |

The default and full grid use group sizes up to 8; larger groups take more
enumeration work. JSON includes the probabilities, gradients, finite-step
reward changes, and support flags. An undefined inverse-ratio HCA control is
`null`, while the marginal-future control remains available.

```pycon
>>> unsupported = evaluate(Config(q0=0, q1=1))
>>> unsupported["return_hca_gradient"] is None
True
>>> abs(unsupported["independent_future_gradient"] - unsupported["true_gradient"]) < 1e-12
True

```

## Reproduce the complete result

```bash
python -m credit_sanity report
python -m credit_sanity verify
python -m unittest discover -s tests -v
python -m doctest docs/walkthrough.md
```

The last command executes the Python examples in this document; CI runs it
on Windows and Ubuntu. The [complete result summary](../results/summary.md)
includes every preset condition. For formula changes, use the independent
checks described in [CONTRIBUTING.md](../CONTRIBUTING.md), and preserve controls
that disagree with the proposed explanation.
