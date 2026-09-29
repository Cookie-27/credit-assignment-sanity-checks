import argparse
from dataclasses import fields
import json
import math
from pathlib import Path

from . import __version__
from .exact import Config, METHODS, evaluate, identical_decision_demo, summarize, sweep
from .reporting import markdown_report


def _compare(actual, expected, path="root"):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Mismatch at {path}: {actual!r} != {expected!r}")
    elif isinstance(expected, (float, int)):
        if type(actual) not in (float, int) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=2e-10):
            raise ValueError(f"Numerical mismatch at {path}: {actual!r} != {expected!r}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"List length mismatch at {path}")
        for index, (a, e) in enumerate(zip(actual, expected)):
            _compare(a, e, f"{path}[{index}]")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Dictionary keys mismatch at {path}")
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}")
    else:
        raise ValueError(f"Unsupported value at {path}")


def _payload():
    rows = sweep()
    return {"version": __version__, "conditions": len(rows), "summary": summarize(rows), "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exact, CPU-only credit-assignment sanity checks.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Evaluate one finite experiment.")
    for name in ("q0", "q1", "p", "gamma", "alpha", "temperature", "epsilon"):
        demo.add_argument("--" + name, type=float, default=getattr(Config(), name))
    demo.add_argument("--group", type=int, default=8)
    demo.add_argument("--echo", action="store_true")
    demo.add_argument("--json", action="store_true")
    full = commands.add_parser("sweep", help="Recompute all 240 fixed theory conditions.")
    full.add_argument("--output", type=Path, default=Path("results/theory.json"))
    verify = commands.add_parser("verify", help="Recompute and compare every committed result.")
    verify.add_argument("--reference", type=Path, default=Path("results/theory.json"))
    report = commands.add_parser("report", help="Recompute the grid and render a readable Markdown summary.")
    report.add_argument("--output", type=Path, help="Write a file instead of printing to stdout.")
    commands.add_parser("coefficient-demo", help="Show cancellation of a same-input winner swap.")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = evaluate(Config(**{field.name: getattr(args, field.name) for field in fields(Config)}))
            if args.json:
                print(json.dumps(result, indent=2, allow_nan=False))
            else:
                print(f"Expected reward: {result['expected_reward']:.9f}")
                print(f"True dJ/dtheta: {result['true_gradient']:+.12f}")
                print(f"{'method':34s} {'expected gradient':>20s} {'reward change (step=.001)':>28s}")
                for name in METHODS:
                    print(f"{name:34s} {result['gradients'][name]:+20.12f} {result['finite_step_reward_changes'][name]:+28.12e}")
                print("Toy first-update diagnostic; not an LLM training result.")
        elif args.command == "sweep":
            payload = _payload()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Wrote {payload['conditions']} exact conditions to {args.output}")
            print(json.dumps(payload["summary"], indent=2))
        elif args.command == "verify":
            expected = json.loads(args.reference.read_text(encoding="utf-8"))
            _compare(_payload(), expected)
            print("PASS: recomputed all 240 conditions and matched the reference.")
        elif args.command == "report":
            report_text = markdown_report(_payload())
            if args.output is None:
                print(report_text, end="")
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report_text, encoding="utf-8")
                print(f"Wrote recomputed summary to {args.output}")
        else:
            print(json.dumps(identical_decision_demo(), indent=2))
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
