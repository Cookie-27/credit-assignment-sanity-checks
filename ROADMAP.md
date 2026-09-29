# Roadmap

The project is a collection of small, executable credit-assignment checks.
Its next additions should help a reader distinguish changes in scores,
coefficients, gradients, and rewards. Items below are directions, not promises
of an algorithmic improvement or a release date.

## Available

- [x] Exact two-step enumeration and the full 240-condition grid.
- [x] Independent ordered-enumeration and objective-gradient checks.
- [x] Complete-echo, equal-value, support-boundary, and same-decision controls.
- [x] Bilingual quick start, reproducible reports, and cross-platform CI.
- [x] An executable walkthrough with Python API examples.

## Small contributions welcome

- Improve an explanation or translate the walkthrough while preserving its
  assumptions and negative controls.
- Propose an informative configuration outside the preset grid. State a
  prediction before interpreting its output and include the full parameters.
- Explain a floating-point or probability-support edge case with a minimal
  reproducer and an independent calculation.

Use the [diagnostic proposal form](https://github.com/Cookie-27/credit-assignment-sanity-checks/issues/new?template=diagnostic.yml)
to discuss a case before expanding the implementation.

## Research extensions under consideration

- [ ] A separate finite environment with two genuinely trainable decisions.
- [ ] More exact support-boundary examples with explicit estimator assumptions.
- [ ] Direct comparisons between score rankings and actual parameter gradients.

Each extension needs a defined objective, a simple baseline, a falsifying
control, and independent numerical validation. More neural-network or agent
experiments would require their own data and compute documentation; the
existing finite results do not establish their outcomes.

## Distribution

Keep the runtime standard-library only and the complete results available.
There is no hosted or paid tier. Useful contributions and independently
reproduced examples are the current participation paths.
