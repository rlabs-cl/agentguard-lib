"""Report formatter — Markdown summary for benchmark results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentguard.benchmark.types import BenchmarkReport, ReadinessScore


def format_report_markdown(report: BenchmarkReport, weights: dict[str, float] | None = None) -> str:
    """Render a BenchmarkReport as a Markdown summary.

    Args:
        report: The benchmark report to format.
        weights: Optional per-dimension relevance weights from the archetype YAML.
            Dimensions with weight=0.0 are rendered as N/A.

    Suitable for CLI output, GitHub comments, or embedding in documentation.
    """
    lines: list[str] = []
    status = "✅ PASSED" if report.overall_passed else "❌ FAILED"

    lines.append(f"# AgentGuard Benchmark Report — {status}")
    lines.append("")
    lines.append(f"**Archetype:** `{report.archetype_id}`")
    lines.append(f"**Model:** `{report.model}`")
    lines.append(f"**Date:** {report.created_at}")
    lines.append(f"**Total Cost:** ${report.total_cost_usd:.4f}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Score | Threshold | Status |")
    lines.append("|--------|------:|----------:|--------|")
    lines.append(
        f"| Enterprise Readiness | {report.enterprise_avg:.3f} | 0.600 | "
        f"{'✅' if report.enterprise_avg >= 0.6 else '❌'} |"
    )
    lines.append(
        f"| Operational Readiness | {report.operational_avg:.3f} | 0.600 | "
        f"{'✅' if report.operational_avg >= 0.6 else '❌'} |"
    )
    lines.append(
        f"| Avg Improvement (Δ) | {report.improvement_avg:+.3f} | +0.050 | "
        f"{'✅' if report.improvement_avg >= 0.05 else '❌'} |"
    )
    lines.append("")

    # Per-complexity results
    lines.append("## Results by Complexity")
    lines.append("")

    for run in report.runs:
        lines.append(f"### {run.complexity.value.title()}")
        lines.append("")
        lines.append(f"> {run.spec[:120]}{'…' if len(run.spec) > 120 else ''}")
        lines.append("")
        lines.append("| | Control (raw LLM) | Treatment (AgentGuard) | Improvement |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| Enterprise | {run.control.enterprise.overall_score:.3f} | "
            f"{run.treatment.enterprise.overall_score:.3f} | "
            f"{run.treatment.enterprise.overall_score - run.control.enterprise.overall_score:+.3f} |"
        )
        lines.append(
            f"| Operational | {run.control.operational.overall_score:.3f} | "
            f"{run.treatment.operational.overall_score:.3f} | "
            f"{run.treatment.operational.overall_score - run.control.operational.overall_score:+.3f} |"
        )
        lines.append(
            f"| Combined | {run.control.combined_score:.3f} | "
            f"{run.treatment.combined_score:.3f} | "
            f"**{run.improvement:+.3f}** |"
        )
        lines.append(
            f"| Files | {run.control.files_generated} | "
            f"{run.treatment.files_generated} | |"
        )
        lines.append(
            f"| Lines | {run.control.total_lines} | "
            f"{run.treatment.total_lines} | |"
        )
        lines.append(
            f"| Cost | ${run.control.cost_usd:.4f} | "
            f"${run.treatment.cost_usd:.4f} | |"
        )
        lines.append("")

        # Side-by-side dimension breakdown (control vs treatment)
        _add_dimension_comparison(lines, "Enterprise", run.control.enterprise, run.treatment.enterprise, weights)
        _add_dimension_comparison(lines, "Operational", run.control.operational, run.treatment.operational, weights)

        # Errors
        if run.control.error:
            lines.append(f"⚠️ **Control error:** {run.control.error}")
            lines.append("")
        if run.treatment.error:
            lines.append(f"⚠️ **Treatment error:** {run.treatment.error}")
            lines.append("")

    # Signature
    if report.signature:
        lines.append("---")
        lines.append(f"*Signed: `{report.signature[:16]}…`*")

    return "\n".join(lines)


def format_report_compact(report: BenchmarkReport) -> str:
    """Render a compact single-line summary for CLI progress output."""
    status = "PASS" if report.overall_passed else "FAIL"
    return (
        f"[{status}] {report.archetype_id} | "
        f"enterprise={report.enterprise_avg:.3f} "
        f"operational={report.operational_avg:.3f} "
        f"Δ={report.improvement_avg:+.3f} "
        f"cost=${report.total_cost_usd:.4f} "
        f"({len(report.runs)} runs)"
    )


def _add_dimension_comparison(
    lines: list[str],
    title: str,
    control: ReadinessScore,
    treatment: ReadinessScore,
    weights: dict[str, float] | None = None,
) -> None:
    """Add a side-by-side control vs treatment per-dimension table.

    Dimensions with ``weight=0.0`` are rendered as N/A and excluded from
    the pass/fail count — they are architecturally irrelevant for the
    archetype being benchmarked.
    """
    ctrl_map = {d.dimension: d for d in control.dimensions}
    treat_map = {d.dimension: d for d in treatment.dimensions}
    all_dims = list(ctrl_map) or list(treat_map)
    if not all_dims:
        return

    # Count applicable (non-zero-weight) dimensions
    applicable = [d for d in all_dims if (weights or {}).get(d, 1.0) > 0.0] if weights else all_dims

    lines.append(
        f"<details><summary>{title} Dimensions — "
        f"Control {control.overall_score:.3f} vs Treatment {treatment.overall_score:.3f}"
        + (f" ({len(all_dims) - len(applicable)} N/A)" if weights and len(applicable) < len(all_dims) else "")
        + "</summary>"
    )
    lines.append("")
    lines.append("| Dimension | Ctrl | Treat | Δ | Pass? | Winner | Findings (treatment) |")
    lines.append("|-----------|-----:|------:|--:|:-----:|:------:|---------------------|")
    for dim_name in all_dims:
        w = (weights or {}).get(dim_name, 1.0)
        if w == 0.0:
            lines.append(
                f"| {dim_name} | — | — | — | N/A | — | *Not applicable for this archetype* |"
            )
            continue
        cd = ctrl_map.get(dim_name)
        td = treat_map.get(dim_name)
        ctrl_score = cd.score if cd else 0.0
        treat_score = td.score if td else 0.0
        delta = treat_score - ctrl_score
        pass_icon = "✅" if (td and td.passed) else "❌"
        if delta > 0.005:
            winner = "treat"
        elif delta < -0.005:
            winner = "ctrl"
        else:
            winner = "tie"
        findings_str = "; ".join(td.findings[:2]) if td and td.findings else "—"
        lines.append(
            f"| {dim_name} | {ctrl_score:.3f} | {treat_score:.3f} | "
            f"{delta:+.3f} | {pass_icon} | {winner} | {findings_str} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")


def _add_dimension_details(
    lines: list[str], title: str, readiness: ReadinessScore,
) -> None:
    """Add dimension-level detail table for a single side (kept for compatibility)."""
    if not readiness.dimensions:
        return

    lines.append(f"<details><summary>{title} ({readiness.overall_score:.3f})</summary>")
    lines.append("")
    lines.append("| Dimension | Score | Status | Findings |")
    lines.append("|-----------|------:|--------|----------|")
    for dim in readiness.dimensions:
        status = "✅" if dim.passed else "❌"
        findings_str = "; ".join(dim.findings[:3]) if dim.findings else "—"
        lines.append(f"| {dim.dimension} | {dim.score:.3f} | {status} | {findings_str} |")
    lines.append("")
    lines.append("</details>")
    lines.append("")
