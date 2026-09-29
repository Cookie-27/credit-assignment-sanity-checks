"""Finite two-step experiment, evaluated by multinomial counts.

This is an independent idealized implementation of published operations,
not the HCAPO authors' code or a reproduction of their agent experiments.
All posterior probabilities are exact. Only the first binary action is
trainable; the second action is a deterministic submit operation.
"""

from dataclasses import asdict, dataclass
from itertools import product
import math

METHODS = (
    "grpo",
    "trajectory_raw",
    "trajectory_smooth",
    "trajectory_raw_unprotected",
    "trajectory_smooth_unprotected",
)
REWARD_PAIRS = (
    (0.5, 0.5), (0.8, 0.8), (1.0, 1.0), (0.51, 0.5),
    (0.9, 0.8), (1.0, 0.99), (0.9, 0.5), (0.25, 0.75),
)
INITIAL_PROBABILITIES = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass(frozen=True)
class Config:
    q0: float = 1.0
    q1: float = 0.99
    p: float = 0.9
    group: int = 8
    echo: bool = False
    gamma: float = 0.95
    alpha: float = 0.5
    temperature: float = 5.0
    epsilon: float = 1e-8

    def __post_init__(self):
        for name in ("q0", "q1", "p", "gamma", "alpha", "temperature", "epsilon"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("q0", "q1", "gamma", "alpha"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.p < 1:
            raise ValueError("p must be strictly between 0 and 1")
        if type(self.group) is not int or not 1 <= self.group <= 32:
            raise ValueError("group must be an integer from 1 through 32")
        if self.temperature <= 0 or self.epsilon <= 0:
            raise ValueError("temperature and epsilon must be positive")
        if type(self.echo) is not bool:
            raise ValueError("echo must be a boolean")


def _sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def reward_at_logit(theta, q0, q1):
    p = _sigmoid(theta)
    return p * q0 + (1 - p) * q1


def outcome_table(config):
    """Four outcomes ordered (a=0,R=0), (0,1), (1,0), (1,1)."""
    pi = (config.p, 1 - config.p)
    q = (config.q0, config.q1)
    marginal_success = pi[0] * q[0] + pi[1] * q[1]
    result = []
    for action, reward in product(range(2), range(2)):
        likelihood = q[action] if reward else 1 - q[action]
        marginal = marginal_success if reward else 1 - marginal_success
        posterior = pi[action] * likelihood / marginal if marginal else 0.0
        if config.echo:
            posterior = 1.0
        score = posterior ** (1 / config.temperature)
        # Submit has posterior 1. Its score is therefore 1 as well.
        denominator = (score + 1) / 2
        rho0 = min(1.2, max(0.8, score / denominator))
        rho1 = min(1.2, max(0.8, 1 / denominator))
        root = config.gamma * rho0 * reward
        final = rho1 * reward
        smoothed = config.alpha * root + (1 - config.alpha) * final
        result.append({
            "action": action, "reward": reward,
            "mass": pi[action] * likelihood,
            "log_policy_gradient": (1.0 if action == 0 else 0.0) - config.p,
            "posterior": posterior, "root_rho": rho0, "final_rho": rho1,
            "root_q": root, "smoothed_root_q": smoothed,
        })
    return result


def _compositions(total):
    for a in range(total + 1):
        for b in range(total - a + 1):
            for c in range(total - a - b + 1):
                yield (a, b, c, total - a - b - c)


def _weighted_z(values, counts, total, epsilon):
    mean = math.fsum(n * value for n, value in zip(counts, values)) / total
    variance = math.fsum(n * (v - mean) ** 2 for n, v in zip(counts, values)) / total
    scale = math.sqrt(variance) + epsilon
    return [(value - mean) / scale for value in values]


def _hca_controls(config):
    """Known reward-conditioned HCA identities, not new estimators.

    The return-conditioned expression is evaluated on the sampled action's
    outcome. The positive-ratio expression uses an independent marginal
    outcome and sums over all actions. Both target the unregularized reward.
    These controls remain reward-conditioned even when echo=True. The
    inverse-ratio control needs overlap on positive-reward outcomes; if one
    action can never succeed, return None instead of silently treating its
    undefined inverse ratio as zero. The general HCA theorem states a
    stronger overlap assumption covering every possible return.
    """
    pi = (config.p, 1 - config.p)
    q = (config.q0, config.q1)
    expected = pi[0] * q[0] + pi[1] * q[1]
    inverse = 0.0
    independent = 0.0
    for a in range(2):
        policy_score = (1.0 if a == 0 else 0.0) - config.p
        if q[a] > 0:
            posterior = pi[a] * q[a] / expected
            inverse += pi[a] * q[a] * policy_score * (1 - pi[a] / posterior)
        if expected > 0:
            posterior = pi[a] * q[a] / expected
            independent += expected * pi[a] * policy_score * posterior / pi[a]
    positive_reward_support = expected == 0 or all(value > 0 for value in q)
    full_support = positive_reward_support and (expected == 1 or all(value < 1 for value in q))
    return inverse if positive_reward_support else None, independent, positive_reward_support, full_support


def evaluate(config):
    outcomes = outcome_table(config)
    n = config.group
    factorials = [math.factorial(i) for i in range(n + 1)]
    rewards = [o["reward"] for o in outcomes]
    raw_values = [o["root_q"] for o in outcomes]
    smooth_values = [o["smoothed_root_q"] for o in outcomes]
    score = [o["log_policy_gradient"] for o in outcomes]
    contributions = {name: [] for name in METHODS}
    probabilities = []
    for counts in _compositions(n):
        coefficient = factorials[n] // math.prod(factorials[k] for k in counts)
        probability = coefficient * math.prod(o["mass"] ** k for o, k in zip(outcomes, counts))
        probabilities.append(probability)
        if probability == 0:
            continue
        macro = _weighted_z(rewards, counts, n, config.epsilon)
        micro = _weighted_z(raw_values, counts, n, config.epsilon)
        smooth = _weighted_z(smooth_values, counts, n, config.epsilon)
        protect = lambda x: [max(v, 0.0) if r else v for v, r in zip(x, rewards)]
        add = lambda x: [a + b for a, b in zip(macro, x)]
        advantages = (macro, add(protect(micro)), add(protect(smooth)), add(micro), add(smooth))
        for name, advantage in zip(METHODS, advantages):
            gradient = math.fsum(k * s * a for k, s, a in zip(counts, score, advantage)) / n
            contributions[name].append(probability * gradient)
    gradients = {name: math.fsum(terms) for name, terms in contributions.items()}
    truth = config.p * (1 - config.p) * (config.q0 - config.q1)
    theta = math.log(config.p / (1 - config.p))
    inverse, independent, reward_support, full_support = _hca_controls(config)
    # Compute changes from delta-p to avoid cancellation for equal rewards.
    changes = {name: (_sigmoid(theta + 1e-3 * grad) - config.p) * (config.q0 - config.q1)
               for name, grad in gradients.items()}
    return {
        "config": asdict(config),
        "true_gradient": truth,
        "expected_reward": config.p * config.q0 + (1 - config.p) * config.q1,
        "gradients": gradients,
        "finite_step_reward_changes": changes,
        "finite_step_size": 1e-3,
        "return_hca_gradient": inverse,
        "return_hca_positive_reward_support": reward_support,
        "return_hca_full_return_support": full_support,
        "independent_future_gradient": independent,
        "probability_sum": math.fsum(probabilities),
        "multinomial_terms": len(probabilities),
        "outcomes": outcomes,
    }


def sweep():
    """The full fixed theory grid: 8 x 5 x 3 x 2 = 240 conditions."""
    return [evaluate(Config(q0=q0, q1=q1, p=p, group=g, echo=echo))
            for q0, q1 in REWARD_PAIRS
            for p in INITIAL_PROBABILITIES
            for g in (2, 4, 8)
            for echo in (False, True)]


def summarize(rows):
    summary = []
    for echo in (False, True):
        subset = [r for r in rows if r["config"]["echo"] == echo]
        nonzero = [r for r in subset if abs(r["true_gradient"]) > 1e-12]
        tied = [r for r in subset if abs(r["true_gradient"]) <= 1e-12]
        for method in METHODS:
            summary.append({
                "echo": echo, "method": method,
                "nonzero_cases": len(nonzero),
                "reversals": sum(r["true_gradient"] * r["gradients"][method] < -1e-12 for r in nonzero),
                "tied_cases": len(tied),
                "tied_drift": sum(abs(r["gradients"][method]) > 1e-10 for r in tied),
            })
    return summary


def identical_decision_demo(epsilon=1e-8):
    """A winner swap need not change an identical decision's total coefficient."""
    def protected_pair(values):
        mean = sum(values) / 2
        scale = abs(values[0] - values[1]) / 2 + epsilon
        return [max(0.0, (value - mean) / scale) for value in values]
    before = protected_pair((0.99, 1.01))
    after = protected_pair((1.01, 0.99))
    return {
        "before": before, "after": after,
        "sum_before": sum(before), "sum_after": sum(after),
        "assumption": "The two samples have identical policy inputs and target tokens.",
        "scope": "Coefficient identity only; this does not equate distinct neural-network gradients.",
    }
