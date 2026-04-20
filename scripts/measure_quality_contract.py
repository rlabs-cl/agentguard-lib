"""Measure the cost of embedding ``quality_contract`` in pipeline responses.

Usage:
    python scripts/measure_quality_contract.py

Produces a per-archetype table with:
    - criteria_count        — number of self-challenge criteria
    - contract_bytes        — size of the quality_contract JSON payload
    - response_bytes_total  — size of the full skeleton response
    - overhead_pct          — contract / (total - contract)

Also emits a summary across all built-in archetypes, and (when available)
the same stats for marketplace archetypes the local AGENTGUARD_API_KEY can
reach. Designed to be reproducible — no randomness, no LLM calls, no
external I/O beyond archetype loading.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys

from agentguard.archetypes.registry import get_archetype_registry
from agentguard.mcp.agent_tools import agentguard_skeleton


# Synthetic spec is archetype-agnostic on purpose — the payload we measure is
# the contract, not the spec rendering.
SPEC = "Synthetic probe for measuring pipeline response overhead."


async def measure_one(archetype_id: str) -> dict:
    raw = await agentguard_skeleton(spec=SPEC, archetype=archetype_id)
    data = json.loads(raw)
    qc = data.get("quality_contract") or {}
    total_bytes = len(raw.encode("utf-8"))
    contract_bytes = len(json.dumps(qc, ensure_ascii=False).encode("utf-8"))
    criteria = qc.get("must_satisfy_criteria", [])
    overhead_pct = (
        100.0 * contract_bytes / max(total_bytes - contract_bytes, 1)
    )
    return {
        "archetype": archetype_id,
        "criteria_count": len(criteria),
        "contract_bytes": contract_bytes,
        "response_bytes_total": total_bytes,
        "overhead_pct": round(overhead_pct, 1),
    }


async def main() -> int:
    registry = get_archetype_registry()
    ids = sorted(registry.list_available())
    if not ids:
        print("No archetypes available locally.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    errors: list[tuple[str, str]] = []
    for aid in ids:
        try:
            rows.append(await measure_one(aid))
        except Exception as exc:  # noqa: BLE001
            errors.append((aid, f"{type(exc).__name__}: {exc}"))

    # Per-archetype table
    print(
        f"{'archetype':<28}{'criteria':>10}{'contract_B':>13}"
        f"{'total_B':>10}{'overhead_%':>13}"
    )
    print("-" * 74)
    for r in rows:
        print(
            f"{r['archetype']:<28}{r['criteria_count']:>10}"
            f"{r['contract_bytes']:>13}{r['response_bytes_total']:>10}"
            f"{r['overhead_pct']:>13}"
        )

    if errors:
        print("\nErrors:")
        for aid, msg in errors:
            print(f"  {aid}: {msg}")

    # Summary
    if rows:
        n = len(rows)
        avg_crit = statistics.mean(r["criteria_count"] for r in rows)
        avg_contract_bytes = statistics.mean(r["contract_bytes"] for r in rows)
        avg_total_bytes = statistics.mean(r["response_bytes_total"] for r in rows)
        avg_overhead = statistics.mean(r["overhead_pct"] for r in rows)
        tokens_contract_approx = avg_contract_bytes / 4.0  # rough GPT tokens

        print("\nSummary (n={}):".format(n))
        print(f"  avg criteria per archetype:   {avg_crit:.1f}")
        print(f"  avg contract bytes:           {avg_contract_bytes:.0f}")
        print(f"  avg total response bytes:     {avg_total_bytes:.0f}")
        print(f"  avg overhead % vs rest:       {avg_overhead:.1f}%")
        print(f"  avg contract tokens (~B/4):   {tokens_contract_approx:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
