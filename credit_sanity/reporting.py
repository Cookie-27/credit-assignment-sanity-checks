"""Readable, deterministic output for the full finite experiment."""


def markdown_report(payload):
    lines = [
        "# Exact experiment results",
        "",
        "Generated with `python -m credit_sanity report --output results/summary.md`.",
        "The command recomputes the experiment; it does not summarize a cached file.",
        "",
        f"Version: {payload['version']}. Conditions: {payload['conditions']}.",
        "Eight reward pairs x five policy probabilities x three group sizes x two",
        "outcome representations. No random sampling, model calls, or training.",
        "",
    ]
    for echo, title in ((False, "Reward-only outcome"), (True, "Complete action echo control")):
        lines.extend([
            f"## {title}", "",
            "| Signal | Opposite directions / unequal-value cases | Drift / equal-value cases |",
            "|---|---:|---:|",
        ])
        for row in payload["summary"]:
            if row["echo"] == echo:
                lines.append(
                    f"| `{row['method']}` | {row['reversals']} / {row['nonzero_cases']} "
                    f"| {row['tied_drift']} / {row['tied_cases']} |"
                )
        lines.append("")
    lines.extend([
        "## Interpretation", "",
        "- Opposite direction means `true_gradient * proxy_gradient < -1e-12`.",
        "- Equal-value drift means `abs(proxy_gradient) > 1e-10` when",
        "  `abs(true_gradient) <= 1e-12`; the reward is constant in this case.",
        "- These are counts in a fixed diagnostic grid, not failure-rate estimates",
        "  for real agents. They concern the expected first update, not learning curves.",
        "- The complete-echo control removes all counted reversals and tied drift",
        "  in this grid. Preserve this result when discussing the counterexamples.",
        "- Individual values, configurations, probability sums, and HCA support",
        "  flags are recorded in [theory.json](theory.json).", "",
        "See the [research note](../docs/research_note.md) for the derivation and",
        "[reproducibility guide](../REPRODUCIBILITY.md) for independent checks.", "",
    ])
    return "\n".join(lines)
