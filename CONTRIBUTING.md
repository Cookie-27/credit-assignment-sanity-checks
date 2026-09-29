# Contributing a useful diagnostic

Small, checkable experiments and corrections are welcome. Start with an issue
or a pull request that states the question, the assumptions, and a command
someone else can run on a CPU.

Start with the [walkthrough](docs/walkthrough.md). Small, scoped contribution
ideas are listed in the [roadmap](ROADMAP.md). Use the issue templates for a
reproduction problem or a scientific correction; a clear counterexample is
a useful contribution even when it rules out a proposed method.

## Adding an experiment

1. Define the environment, objective, estimator, and evaluation quantity.
2. Include a simple baseline and a control that could disprove the explanation.
3. Report the full declared grid, including negative results. Explain any
   cases excluded because the estimator is undefined.
4. Validate the numerical implementation through a different calculation,
   a known identity, or a finite-difference check.
5. Separate score changes, update coefficients, actual gradients, and measured
   reward changes. Specify which of these the experiment measures.

Keep the core standard-library only. If an extension needs models or external
environments, discuss its scope before adding dependencies. Attribute borrowed
formulas and code, and document their licenses.

## Before submitting

From the repository root:

```bash
python -m unittest discover -s tests -v
python -m credit_sanity verify
python -m doctest docs/walkthrough.md
```

When deliberately changing the experiment, regenerate both artifacts and
explain the scientific reason for each changed result:

```bash
python -m credit_sanity sweep --output results/theory.json
python -m credit_sanity report --output results/summary.md
```

For a bug report, include Python/OS versions, the command, expected behavior,
and actual output. For a scientific correction, point to the relevant formula
or assumption and, where possible, provide a minimal counterexample.

This repository makes no paper-novelty or general LLM-performance claim. A new
negative example should be described at the scope supported by its evidence.
