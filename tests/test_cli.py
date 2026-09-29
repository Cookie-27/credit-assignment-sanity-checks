"""Exercise the user-facing entry point and result-integrity failures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-B", "-S", "-m", "credit_sanity", *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        timeout=60,
    )


class CliTests(unittest.TestCase):
    def test_json_demo_and_support_boundary(self):
        command = run_cli("demo", "--q0", "0", "--q1", "1", "--json")
        self.assertEqual(command.returncode, 0, command.stderr)
        result = json.loads(command.stdout)
        self.assertIsNone(result["return_hca_gradient"])
        self.assertLess(result["true_gradient"], 0)

    def test_invalid_probability_is_a_cli_error(self):
        command = run_cli("demo", "--p", "1")
        self.assertEqual(command.returncode, 2)
        self.assertIn("p must be strictly between 0 and 1", command.stderr)

    def test_verify_rejects_corruption(self):
        reference = json.loads((ROOT / "results/theory.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "corrupted.json"
            # A boolean must not silently compare equal to the numeric value 1.
            reference["rows"][0]["probability_sum"] = True
            path.write_text(json.dumps(reference), encoding="utf-8")
            command = run_cli("verify", "--reference", str(path))
            self.assertEqual(command.returncode, 2)
            self.assertIn("root.rows[0].probability_sum", command.stderr)

    def test_report_has_both_controls_and_reproduces_committed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "summary.md"
            command = run_cli("report", "--output", str(path))
            self.assertEqual(command.returncode, 0, command.stderr)
            report = path.read_text(encoding="utf-8")
        self.assertIn("Conditions: 240.", report)
        self.assertIn("| `trajectory_smooth` | 23 / 75 | 36 / 45 |", report)
        self.assertIn("| `trajectory_smooth` | 0 / 75 | 0 / 45 |", report)
        self.assertEqual(report, (ROOT / "results/summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
