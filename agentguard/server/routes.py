"""REST route handlers for the AgentGuard HTTP service.

All routes live under the ``/v1`` prefix (added by ``app.include_router``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agentguard.server.schemas import (
    ArchetypeDetail,
    ArchetypeSummary,
    ArchetypeVerifyRequest,
    ArchetypeVerifyResponse,
    AutoFixResponse,
    ChallengeRequest,
    ChallengeResponse,
    CheckResultResponse,
    CriterionResultResponse,
    GenerateRequest,
    GenerateResponse,
    TraceDetail,
    TraceListItem,
    TraceSummaryResponse,
    ValidateRequest,
    ValidateResponse,
    ValidationErrorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _trace_store(request: Request) -> str | None:
    """Return the trace store directory from app state."""
    return getattr(request.app.state, "trace_store", None)


def _trace_summary_from_trace(trace: Any) -> TraceSummaryResponse:
    """Convert a Trace object to a TraceSummaryResponse."""
    summary = trace.summary()
    return TraceSummaryResponse(
        id=trace.id,
        total_llm_calls=summary.total_llm_calls,
        total_cost={
            "total": str(summary.total_cost.total_cost),
            "currency": "USD",
        },
        structural_fixes=summary.structural_fixes,
        challenge_reworks=summary.challenge_reworks,
        duration_ms=summary.duration_ms,
        levels_completed=summary.levels_completed,
    )


# ------------------------------------------------------------------ #
#  POST /v1/generate
# ------------------------------------------------------------------ #


@router.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    """Run the full top-down generation pipeline."""
    from agentguard.pipeline import Pipeline

    try:
        pipe = Pipeline(
            archetype=body.archetype,
            llm=body.llm,
            trace_store=_trace_store(request),
        )
        result = await pipe.generate(
            body.spec,
            skip_challenge=body.options.skip_challenge,
            skip_validation=body.options.skip_validation,
            parallel_l4=body.options.parallel_l4,
            max_challenge_retries=body.options.max_challenge_retries,
        )

        trace_resp = TraceSummaryResponse()
        if result.trace:
            trace_resp = _trace_summary_from_trace(result.trace)

        return GenerateResponse(files=result.files, trace=trace_resp)

    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------ #
#  POST /v1/validate
# ------------------------------------------------------------------ #


@router.post("/validate", response_model=ValidateResponse, tags=["quality"])
async def validate(body: ValidateRequest) -> ValidateResponse:
    """Run structural validation on code."""
    from agentguard.archetypes.base import Archetype
    from agentguard.validation.validator import Validator

    try:
        archetype = None
        if body.archetype:
            archetype = Archetype.load(body.archetype)

        validator = Validator(archetype=archetype)
        report = validator.check(
            body.files,
            checks=body.checks,
        )

        checks = [
            CheckResultResponse(
                check=c.check,
                passed=c.passed,
                details=c.details,
                duration_ms=c.duration_ms,
            )
            for c in report.checks
        ]
        auto_fixed = [
            AutoFixResponse(
                file=f.file_path,
                fix=f.description,
            )
            for f in report.auto_fixed
        ]
        errors = [
            ValidationErrorResponse(
                check=e.check,
                file_path=e.file_path,
                line=e.line,
                column=e.column,
                message=e.message,
                severity=e.severity.value,
                code=e.code,
            )
            for e in report.errors
        ]

        return ValidateResponse(
            passed=report.passed,
            checks=checks,
            auto_fixed=auto_fixed,
            errors=errors,
        )

    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Validation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------ #
#  POST /v1/challenge
# ------------------------------------------------------------------ #


@router.post("/challenge", response_model=ChallengeResponse, tags=["quality"])
async def challenge(body: ChallengeRequest) -> ChallengeResponse:
    """Run self-challenge evaluation on code."""
    from agentguard.challenge.challenger import SelfChallenger
    from agentguard.llm.factory import create_llm_provider

    try:
        llm = create_llm_provider(body.llm)
        challenger = SelfChallenger(llm=llm)

        result = await challenger.challenge(
            output=body.code,
            criteria=body.criteria or [],
            task_description="Code review via API",
        )

        criteria_results = [
            CriterionResultResponse(
                criterion=c.criterion,
                passed=c.passed,
                explanation=c.explanation,
            )
            for c in result.criteria_results
        ]

        return ChallengeResponse(
            passed=result.passed,
            attempt=result.attempt,
            criteria_results=criteria_results,
            assumptions=result.assumptions,
            grounding_violations=result.grounding_violations,
            feedback=result.feedback,
            cost={
                "total": str(result.cost.total_cost),
                "currency": "USD",
            },
        )

    except Exception as exc:
        logger.exception("Challenge failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------ #
#  Individual generation levels
# ------------------------------------------------------------------ #


@router.post("/skeleton", tags=["generation"])
async def skeleton(body: GenerateRequest, request: Request) -> dict[str, Any]:
    """Run L1 skeleton generation only."""
    from agentguard.pipeline import Pipeline

    try:
        pipe = Pipeline(
            archetype=body.archetype,
            llm=body.llm,
            trace_store=_trace_store(request),
        )
        result = await pipe.skeleton(body.spec)
        return {
            "files": [{"path": f.path, "purpose": f.purpose} for f in result.files],
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Skeleton generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/contracts", tags=["generation"])
async def contracts(body: GenerateRequest, request: Request) -> dict[str, Any]:
    """Run L1 + L2 (skeleton → contracts)."""
    from agentguard.pipeline import Pipeline

    try:
        pipe = Pipeline(
            archetype=body.archetype,
            llm=body.llm,
            trace_store=_trace_store(request),
        )
        skel = await pipe.skeleton(body.spec)
        result = await pipe.contracts(body.spec, skel)
        return {"files": result.files}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Contracts generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/wiring", tags=["generation"])
async def wiring(body: GenerateRequest, request: Request) -> dict[str, Any]:
    """Run L1 → L2 → L3 (through wiring)."""
    from agentguard.pipeline import Pipeline

    try:
        pipe = Pipeline(
            archetype=body.archetype,
            llm=body.llm,
            trace_store=_trace_store(request),
        )
        skel = await pipe.skeleton(body.spec)
        cont = await pipe.contracts(body.spec, skel)
        result = await pipe.wiring(cont)
        return {"files": result.files}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Wiring generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/logic", tags=["generation"])
async def logic(body: GenerateRequest, request: Request) -> dict[str, Any]:
    """Run L1 → L2 → L3 → L4 (through logic)."""
    from agentguard.pipeline import Pipeline

    try:
        pipe = Pipeline(
            archetype=body.archetype,
            llm=body.llm,
            trace_store=_trace_store(request),
        )
        skel = await pipe.skeleton(body.spec)
        cont = await pipe.contracts(body.spec, skel)
        wir = await pipe.wiring(cont)
        result = await pipe.logic(wir)
        return {"files": result.files}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Logic generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------ #
#  GET /v1/archetypes
# ------------------------------------------------------------------ #


@router.get(
    "/archetypes",
    response_model=list[ArchetypeSummary],
    tags=["archetypes"],
)
async def list_archetypes() -> list[ArchetypeSummary]:
    """List all available archetypes."""
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    result = []
    for arch_id in registry.list_available():
        entry = registry.get_entry(arch_id)
        result.append(
            ArchetypeSummary(
                id=entry.archetype.id,
                name=entry.archetype.name,
                description=entry.archetype.description,
                trust_level=entry.trust_level.value,
                content_hash=entry.content_hash,
            )
        )
    return result


@router.get(
    "/archetypes/{archetype_id}",
    response_model=ArchetypeDetail,
    tags=["archetypes"],
)
async def get_archetype(archetype_id: str) -> ArchetypeDetail:
    """Get detailed archetype configuration."""
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    try:
        entry = registry.get_entry(archetype_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    arch = entry.archetype
    return ArchetypeDetail(
        id=arch.id,
        name=arch.name,
        description=arch.description,
        version=arch.version,
        trust_level=entry.trust_level.value,
        content_hash=entry.content_hash,
        tech_stack={
            "language": arch.tech_stack.language,
            "framework": arch.tech_stack.framework,
            "database": arch.tech_stack.database,
            "testing": arch.tech_stack.testing,
            "linter": arch.tech_stack.linter,
            "type_checker": arch.tech_stack.type_checker,
        },
        pipeline={
            "levels": arch.pipeline.levels,
            "enable_self_challenge": arch.pipeline.enable_self_challenge,
            "enable_structural_validation": arch.pipeline.enable_structural_validation,
            "max_self_challenge_retries": arch.pipeline.max_self_challenge_retries,
        },
        structure=arch.structure,
        validation={
            "checks": arch.validation.checks,
            "lint_rules": arch.validation.lint_rules,
            "type_strictness": arch.validation.type_strictness,
        },
        self_challenge={
            "criteria": arch.self_challenge.criteria,
            "grounding_check": arch.self_challenge.grounding_check,
            "assumptions_must_declare": arch.self_challenge.assumptions_must_declare,
        },
        reference_patterns=arch.reference_patterns,
    )


@router.post(
    "/archetypes/verify",
    response_model=ArchetypeVerifyResponse,
    tags=["archetypes"],
)
async def verify_archetype(body: ArchetypeVerifyRequest) -> ArchetypeVerifyResponse:
    """Verify archetype integrity against the registry.

    Check if an archetype is registered and optionally verify a content hash.
    This is the source-of-truth endpoint for integrity verification.
    """
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    if not registry.is_registered(body.archetype_id):
        return ArchetypeVerifyResponse(
            archetype_id=body.archetype_id,
            registered=False,
        )

    entry = registry.get_entry(body.archetype_id)
    hash_match = None
    if body.content_hash:
        hash_match = entry.content_hash == body.content_hash

    return ArchetypeVerifyResponse(
        archetype_id=body.archetype_id,
        registered=True,
        trust_level=entry.trust_level.value,
        content_hash=entry.content_hash,
        hash_match=hash_match,
    )


# ------------------------------------------------------------------ #
#  GET /v1/traces
# ------------------------------------------------------------------ #


@router.get("/traces", response_model=list[TraceListItem], tags=["traces"])
async def list_traces(request: Request) -> list[TraceListItem]:
    """List recent traces from the trace store."""
    store = _trace_store(request)
    if not store:
        return []

    store_path = Path(store)
    if not store_path.exists():
        return []

    traces: list[TraceListItem] = []
    for trace_file in sorted(store_path.glob("*.json"), reverse=True)[:50]:
        try:
            data = json.loads(trace_file.read_text(encoding="utf-8"))
            traces.append(
                TraceListItem(
                    id=data.get("id", trace_file.stem),
                    archetype=data.get("archetype", ""),
                    spec_summary=data.get("spec_summary", ""),
                    started_at=data.get("started_at", ""),
                    total_cost=str(data.get("total_cost", "0")),
                    duration_ms=data.get("duration_ms", 0),
                )
            )
        except (json.JSONDecodeError, OSError):
            continue

    return traces


@router.get("/traces/{trace_id}", response_model=TraceDetail, tags=["traces"])
async def get_trace(trace_id: str, request: Request) -> TraceDetail:
    """Get a specific trace by ID."""
    store = _trace_store(request)
    if not store:
        raise HTTPException(status_code=404, detail="No trace store configured")

    trace_file = Path(store) / f"{trace_id}.json"
    if not trace_file.exists():
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found")

    try:
        data = json.loads(trace_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Failed to read trace") from exc

    return TraceDetail(
        id=data.get("id", trace_id),
        archetype=data.get("archetype", ""),
        spec_summary=data.get("spec_summary", ""),
        started_at=data.get("started_at", ""),
        ended_at=data.get("ended_at"),
        spans=data.get("spans", []),
        summary=TraceSummaryResponse(
            id=data.get("id", trace_id),
            levels_completed=[],
            duration_ms=data.get("duration_ms", 0),
        ),
    )
