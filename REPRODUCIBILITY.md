# Reproducing version 0.1.0

## Requirements

Python 3.10 or newer. The CLI, experiment, tests, and walkthrough use only
the standard library. The locally
verified environment is Windows with Python 3.12.14. Running from the root
of the extracted repository requires no package installation.

The optional GIF-authoring script uses Pillow only when invoked with `--gif`;
Pillow is not needed for any experiment or verification command.

## Commands

```bash
python -m unittest discover -s tests -v
python -m credit_sanity verify --reference results/theory.json
python -m credit_sanity demo --json
python -m credit_sanity coefficient-demo
python -m credit_sanity report
python -m doctest docs/walkthrough.md
```

To regenerate rather than check the artifact:

```bash
python -m credit_sanity sweep --output results/theory.json
python -m credit_sanity report --output results/summary.md
```

The committed JSON contains the full configuration, exact probability sum,
all method gradients, a finite-step reward change, outcome tables, and control
identities for every condition. Output contains no timestamps or absolute
paths. Very small floating-point differences across Python/platform versions
are allowed by the verifier (`abs_tol=2e-10`, `rel_tol=1e-9`).

## Independent checks

The main code uses multinomial count enumeration. Tests use a separate
implementation that enumerates ordered outcome sequences for smaller groups,
including variable smoothing coefficients and discounts. Further tests check
finite differences of the terminal-reward objective, valid-support HCA
identities, unsupported inverse-ratio cases, the two-step algebra, and
controls that refute overbroad interpretations.

The CI workflow requests Python 3.10 and 3.12 on Windows and Ubuntu. See the
[actual workflow runs](https://github.com/Cookie-27/credit-assignment-sanity-checks/actions/workflows/checks.yml)
for execution status. The CLI tests also recompute the readable Markdown
summary and check it against the committed artifact, reject a corrupted
reference, and exercise invalid-input and support-boundary behavior.

`report` recomputes the complete grid. Its stdout is Markdown; `--output`
writes the same text to a file. There are no timestamps or machine-specific
paths in either generated result artifact.

## Scope

This release reproduces the 240-condition finite theory experiment. It does
not package the larger private research workspace, LLM checkpoints, API
fixtures, real model trajectories, or experiments derived from other
environments. Those are unnecessary to reproduce the claims made here.
