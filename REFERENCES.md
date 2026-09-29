# Primary references

- [Hindsight Credit Assignment for Long-Horizon LLM Agents](https://arxiv.org/html/2603.08754v1).
  HCAPO, 2026. The motivating operations are token-likelihood scoring,
  within-trajectory normalization, macro/micro combination, successful-negative
  protection (§4), and adjacent smoothing (Appendix A). We independently
  implement these operations in an oracle two-step setting. No author source
  code or exact hindsight prompt was used.
- [Hindsight Credit Assignment](https://proceedings.neurips.cc/paper/9413-hindsight-credit-assignment.pdf).
  Harutyunyan et al., NeurIPS 2019. Theorem 1 uses marginal future outcomes;
  Theorem 2 gives the inverse-ratio return expression with an explicit
  posterior-support assumption. These are the known identity controls.
- [Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis](https://arxiv.org/abs/2306.16803).
  COCOA, 2023. Relevant background on outcome representations and spurious
  contribution estimates; this repository does not reproduce its experiments.

Related work also includes [QVal](https://arxiv.org/abs/2606.32034), on evaluating
dense signals against reference action values, and
[FACTOR](https://arxiv.org/abs/2608.07118), on action credit and token allocation.
No reference-Q benchmark or credit-conserving training algorithm is claimed
as new here.

References are links only. Their papers, implementations, datasets, and model
weights are not redistributed in this repository.
