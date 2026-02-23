"""Report formatter — Markdown summary for benchmark results."""

from __future__ import annotations

from agentguard.benchmark.types import BenchmarkReport, ComplexityRun, ReadinessScore


def format_report_markdown(report: BenchmarkReport) -> str:
    """Render a BenchmarkReport as a Markdown summary.

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

        # Dimension breakdowns for treatment
        _add_dimension_details(lines, "Enterprise Dimensions (Treatment)", run.treatment.enterprise)
        _add_dimension_details(lines, "Operational Dimensions (Treatment)", run.treatment.operational)

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


def _add_dimension_details(
    lines: list[str], title: str, readiness: ReadinessScore,
) -> None:
    """Add dimension-level detail table."""
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
