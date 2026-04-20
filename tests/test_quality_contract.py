"""v0.13.0 regression: pipeline tools embed the quality contract upfront.

Prior to 0.13.0 the agent only encountered the self-challenge criteria when
``validate`` ran at the end of the pipeline. For long, multi-section
artefacts (and for content archetypes especially) this caused drift — later
sections were written without the earlier sections' rubric in attention.
These tests pin that every stage of the pipeline returns a
``quality_contract`` block containing the criteria, structure, and
enforcement note, so the generating agent has the rubric before committing
output.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agentguard.mcp.agent_tools import (
    agentguard_contracts_and_wiring,
    agentguard_logic,
    agentguard_skeleton,
)


_CONTRACT_KEYS = {
    "stage",
    "must_satisfy_criteria",
    "expected_structure",
    "validation_checks",
    "enforcement",
}


def _extract_contract(raw: str) -> dict:
    data = json.loads(raw)
    assert "quality_contract" in data, f"quality_contract missing; keys={sorted(data)}"
    qc = data["quality_contract"]
    missing = _CONTRACT_KEYS - set(qc)
    assert not missing, f"quality_contract missing keys: {sorted(missing)}"
    return qc


@pytest.mark.asyncio
async def test_skeleton_embeds_quality_contract() -> None:
    raw = await agentguard_skeleton(
        spec="A minimal REST API with a health endpoint",
        archetype="api_backend",
    )
    qc = _extract_contract(raw)
    assert qc["stage"] == "L1_skeleton"
    # The archetype ships with criteria — if the list is empty the contract
    # is a no-op and the regression is effectively undone. Fail loud.
    assert qc["must_satisfy_criteria"], (
        "api_backend archetype shipped without self-challenge criteria — "
        "either the archetype changed or the loader is dropping them."
    )
    assert qc["expected_structure"]["dirs"] or qc["expected_structure"]["files"]
    assert qc["enforcement"].startswith("Generate this stage")


@pytest.mark.asyncio
async def test_contracts_and_wiring_embeds_quality_contract() -> None:
    skeleton = json.dumps(
        [{"path": "src/main.py", "purpose": "entry point", "tier": "feature"}]
    )
    raw = await agentguard_contracts_and_wiring(
        spec="A minimal REST API with a health endpoint",
        skeleton_json=skeleton,
        archetype="api_backend",
    )
    qc = _extract_contract(raw)
    assert qc["stage"] == "L2_L3_contracts_and_wiring"
    assert qc["must_satisfy_criteria"]


@pytest.mark.asyncio
async def test_logic_embeds_quality_contract() -> None:
    raw = await agentguard_logic(
        file_path="src/main.py",
        file_code="def foo():\n    raise NotImplementedError",
        function_name="foo",
        archetype="api_backend",
    )
    qc = _extract_contract(raw)
    assert qc["stage"] == "L4_logic"
    assert qc["must_satisfy_criteria"]


@pytest.mark.asyncio
async def test_quality_contract_is_consistent_across_stages() -> None:
    """The same archetype should produce the same criteria list at every stage.

    If this test fails, some stage is filtering or transforming the criteria,
    which would defeat the 'rubric in attention throughout' guarantee.
    """
    spec = "A minimal REST API with a health endpoint"
    arch = "api_backend"

    s = _extract_contract(await agentguard_skeleton(spec=spec, archetype=arch))
    cw = _extract_contract(
        await agentguard_contracts_and_wiring(
            spec=spec,
            skeleton_json=json.dumps(
                [{"path": "src/main.py", "purpose": "entry", "tier": "feature"}]
            ),
            archetype=arch,
        )
    )
    l = _extract_contract(  # noqa: E741
        await agentguard_logic(
            file_path="src/main.py",
            file_code="def foo():\n    raise NotImplementedError",
            function_name="foo",
            archetype=arch,
        )
    )

    assert s["must_satisfy_criteria"] == cw["must_satisfy_criteria"] == l["must_satisfy_criteria"]
    assert s["expected_structure"] == cw["expected_structure"] == l["expected_structure"]
    assert s["validation_checks"] == cw["validation_checks"] == l["validation_checks"]
