"""Small independent ordered-batch reference and mathematical controls."""
from itertools import product
import math
import random
import unittest

from credit_sanity.exact import Config, METHODS, evaluate, identical_decision_demo, reward_at_logit, sweep, summarize


def ordered_reference(config):
    # Directly enumerate action/reward sequences. Does not use the production
    # outcome table, count enumeration, normalization, or advantage helpers.
    p, q, g = config.p, (config.q0, config.q1), config.group
    mean_reward = p * q[0] + (1 - p) * q[1]
    result = {name: 0.0 for name in METHODS}
    for sequence in product(range(4), repeat=g):
        actions = [x // 2 for x in sequence]
        rewards = [x % 2 for x in sequence]
        probability = math.prod((p if a == 0 else 1 - p) * (q[a] if r else 1 - q[a]) for a, r in zip(actions, rewards))
        if probability == 0:
            continue
        raw, smooth = [], []
        for a, r in zip(actions, rewards):
            marginal = mean_reward if r else 1 - mean_reward
            h = ((p if a == 0 else 1 - p) * (q[a] if r else 1 - q[a]) / marginal)
            if config.echo:
                h = 1.0
            w = h ** (1 / config.temperature)
            root = min(1.2, max(0.8, 2 * w / (w + 1)))
            final = min(1.2, max(0.8, 2 / (w + 1)))
            raw.append(config.gamma * r * root)
            smooth.append(r * (config.alpha * config.gamma * root + (1 - config.alpha) * final))
        def z(values):
            mean = sum(values) / g
            scale = math.sqrt(sum((v - mean) ** 2 for v in values) / g) + config.epsilon
            return [(v - mean) / scale for v in values]
        macro, micro, smicro = z(rewards), z(raw), z(smooth)
        for name in METHODS:
            value = 0.0
            for i, a in enumerate(actions):
                local = 0.0 if name == "grpo" else (smicro if "smooth" in name else micro)[i]
                if name != "grpo" and "unprotected" not in name and rewards[i]:
                    local = max(local, 0.0)
                value += ((1.0 if a == 0 else 0.0) - p) * (macro[i] + local) / g
            result[name] += probability * value
    return result


class ExactTests(unittest.TestCase):
    def test_ordered_enumeration_agrees(self):
        rng = random.Random(1949)
        for group in (1, 2, 3, 4):
            for echo in (False, True):
                config = Config(q0=rng.random(), q1=rng.random(), p=rng.uniform(.1, .9), group=group, echo=echo,
                                gamma=rng.uniform(.8, 1), alpha=rng.random())
                actual = evaluate(config)
                for name, expected in ordered_reference(config).items():
                    self.assertAlmostEqual(actual["gradients"][name], expected, delta=2e-10)

    def test_hca_and_finite_difference(self):
        for q0, q1 in ((0, 0), (1, 1), (0, 1), (.51, .5), (.9, .1)):
            for p in (.1, .5, .9):
                result = evaluate(Config(q0=q0, q1=q1, p=p))
                truth = result["true_gradient"]
                self.assertAlmostEqual(result["probability_sum"], 1, delta=1e-12)
                if q0 == 0 and q1 > 0 or q1 == 0 and q0 > 0:
                    self.assertIsNone(result["return_hca_gradient"])
                    self.assertFalse(result["return_hca_positive_reward_support"])
                else:
                    self.assertAlmostEqual(result["return_hca_gradient"], truth, delta=1e-12)
                self.assertAlmostEqual(result["independent_future_gradient"], truth, delta=1e-12)
                theta, step = math.log(p / (1 - p)), 1e-5
                numerical = (reward_at_logit(theta + step, q0, q1) - reward_at_logit(theta - step, q0, q1)) / (2 * step)
                self.assertAlmostEqual(numerical, truth, delta=2e-11)

    def test_two_step_identity(self):
        for gamma, alpha in ((.95, .5), (1, .5), (.95, 1 / 1.95), (.9, .7)):
            result = evaluate(Config(gamma=gamma, alpha=alpha))
            for outcome in result["outcomes"]:
                self.assertAlmostEqual(outcome["root_rho"] + outcome["final_rho"], 2, delta=1e-12)
                expected = outcome["reward"] * (2 * (1 - alpha) + (alpha * (1 + gamma) - 1) * outcome["root_rho"])
                self.assertAlmostEqual(outcome["smoothed_root_q"], expected, delta=1e-12)

    def test_known_counterexample(self):
        result = evaluate(Config())
        self.assertAlmostEqual(result["true_gradient"], .0009, delta=1e-14)
        self.assertGreater(result["gradients"]["grpo"], 0)
        self.assertLess(result["gradients"]["trajectory_smooth"], 0)
        self.assertAlmostEqual(result["gradients"]["trajectory_smooth"], -.1809824418, delta=1e-10)

    def test_drift_is_not_reward_harm(self):
        result = evaluate(Config(q0=1, q1=1, p=.9))
        self.assertEqual(result["true_gradient"], 0)
        self.assertLess(result["gradients"]["trajectory_smooth"], -.1)
        self.assertTrue(all(value == 0 for value in result["finite_step_reward_changes"].values()))

    def test_fixed_grid_and_echo_control(self):
        rows = sweep()
        self.assertEqual(len(rows), 240)
        for summary in summarize(rows):
            if summary["echo"]:
                self.assertEqual(summary["reversals"], 0)
                self.assertEqual(summary["tied_drift"], 0)
            elif summary["method"] == "trajectory_smooth":
                self.assertEqual((summary["nonzero_cases"], summary["reversals"]), (75, 23))

    def test_identical_decision_winner_swap(self):
        result = identical_decision_demo()
        self.assertNotEqual(result["before"], result["after"])
        self.assertAlmostEqual(result["sum_before"], result["sum_after"], delta=1e-14)

    def test_invalid_inputs(self):
        for kwargs in ({"p": 0}, {"p": 1}, {"q0": -1}, {"q1": 2}, {"group": 0}, {"group": True},
                       {"group": 33}, {"alpha": 2}, {"temperature": 0}, {"epsilon": 0}, {"p": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
