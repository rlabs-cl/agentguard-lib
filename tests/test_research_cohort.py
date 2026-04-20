"""Tests for the research-cohort telemetry subsystem."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agentguard.stats import collector as collector_mod
from agentguard.stats import db as db_mod
from agentguard.stats import research as research_mod
from agentguard.stats.collector import (
    StatsCollector,
    current_research_cohort,
)
from agentguard.stats.research import (
    anonymise_events,
    get_cohort_events,
    upload_cohort,
)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the stats DB at a throwaway path for test isolation."""
    path = tmp_path / "stats.db"
    monkeypatch.setattr(db_mod, "_DB_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "_DB_PATH", str(path))
    # Reset the module-level singleton so a fresh collector reads our env.
    collector_mod._collector = None
    yield path
    collector_mod._collector = None


def _clear_cohort_env(monkeypatch):
    monkeypatch.delenv("AGENTGUARD_RESEARCH_COHORT", raising=False)


# ── current_research_cohort ─────────────────────────────────────────


def test_current_research_cohort_returns_none_when_unset(monkeypatch):
    _clear_cohort_env(monkeypatch)
    assert current_research_cohort() is None


def test_current_research_cohort_strips_blank(monkeypatch):
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "   ")
    assert current_research_cohort() is None


def test_current_research_cohort_returns_value(monkeypatch):
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "uach-2026-05")
    assert current_research_cohort() == "uach-2026-05"


# ── StatsCollector integration ──────────────────────────────────────


def test_collector_records_null_cohort_when_unset(tmp_db, monkeypatch):
    _clear_cohort_env(monkeypatch)
    c = StatsCollector()
    assert c.research_cohort_id is None
    c.record_tool_call("agentguard_skeleton", duration_ms=100)

    conn = db_mod.get_connection()
    try:
        row = conn.execute(
            "SELECT research_cohort_id FROM tool_events",
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["research_cohort_id"] is None


def test_collector_records_cohort_when_set(tmp_db, monkeypatch):
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "cohort-42")
    c = StatsCollector()
    assert c.research_cohort_id == "cohort-42"
    c.record_tool_call("agentguard_skeleton", duration_ms=100)

    conn = db_mod.get_connection()
    try:
        row = conn.execute(
            "SELECT research_cohort_id FROM tool_events",
        ).fetchone()
    finally:
        conn.close()
    assert row["research_cohort_id"] == "cohort-42"


def test_cohort_snapshot_does_not_follow_mid_session_env_flip(
    tmp_db, monkeypatch,
):
    # Collector resolves cohort at __init__; env changes later must not
    # split events across two cohort ids for the same session.
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "A")
    c = StatsCollector()
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "B")
    c.record_tool_call("agentguard_skeleton", duration_ms=10)

    conn = db_mod.get_connection()
    try:
        cohorts = [r["research_cohort_id"] for r in conn.execute(
            "SELECT research_cohort_id FROM tool_events",
        )]
    finally:
        conn.close()
    assert cohorts == ["A"]


# ── get_cohort_events ───────────────────────────────────────────────


def test_get_cohort_events_filters_by_cohort(tmp_db, monkeypatch):
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "in-cohort")
    StatsCollector().record_tool_call("t1", duration_ms=1)

    monkeypatch.delenv("AGENTGUARD_RESEARCH_COHORT")
    collector_mod._collector = None
    StatsCollector().record_tool_call("t2", duration_ms=2)

    rows_in = get_cohort_events("in-cohort")
    rows_out = get_cohort_events("other-cohort")
    assert len(rows_in) == 1
    assert rows_in[0]["tool_name"] == "t1"
    assert rows_out == []


def test_get_cohort_events_rejects_empty_cohort(tmp_db):
    assert get_cohort_events("") == []


# ── anonymise_events ────────────────────────────────────────────────


def test_anonymise_hashes_paths_and_drops_parameters_summary():
    events = [
        {
            "project_path": "/home/alice/secret-project",
            "project_name": "secret-project",
            "parameters_summary": "spec: build the rocket",
            "tool_name": "agentguard_skeleton",
            "duration_ms": 42,
            "error_message": None,
        },
    ]
    anon = anonymise_events(events)
    assert "parameters_summary" not in anon[0]
    assert anon[0]["project_path"] != "/home/alice/secret-project"
    assert anon[0]["project_name"] != "secret-project"
    assert len(anon[0]["project_path"]) == 12
    assert len(anon[0]["project_name"]) == 12
    # Signal preserved.
    assert anon[0]["tool_name"] == "agentguard_skeleton"
    assert anon[0]["duration_ms"] == 42


def test_anonymise_is_deterministic():
    events = [{"project_path": "/tmp/x", "project_name": "x"}]
    a = anonymise_events(events)
    b = anonymise_events(events)
    assert a[0]["project_path"] == b[0]["project_path"]


def test_anonymise_truncates_error_message():
    events = [{"error_message": "X" * 500, "project_path": None, "project_name": None}]
    anon = anonymise_events(events)
    assert len(anon[0]["error_message"]) == 200


# ── upload_cohort ───────────────────────────────────────────────────


def test_upload_cohort_rejects_empty_cohort_id(tmp_db):
    result = upload_cohort("", dry_run=True)
    assert result["status"] == "error"
    assert result["records_exported"] == 0


def test_upload_cohort_dry_run_returns_payload_without_sending(tmp_db, monkeypatch):
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "dry-cohort")
    StatsCollector().record_tool_call("agentguard_skeleton", duration_ms=5)

    # Guard: _post_payload must not be called on dry_run.
    with patch.object(research_mod, "_post_payload") as post:
        result = upload_cohort("dry-cohort", dry_run=True)
        post.assert_not_called()

    assert result["status"] == "dry_run"
    assert result["records_exported"] == 1
    assert "payload" in result
    # Anonymisation was applied to the payload.
    ev = result["payload"]["events"][0]
    assert "parameters_summary" not in ev


def test_upload_cohort_happy_path_posts_once(tmp_db, monkeypatch):
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "live-cohort")
    StatsCollector().record_tool_call("agentguard_skeleton", duration_ms=5)

    with patch.object(research_mod, "_post_payload") as post:
        post.return_value = {
            "records_exported": 1,
            "destination": "https://example/research",
            "status": "success",
            "cohort_id": "live-cohort",
        }
        result = upload_cohort("live-cohort", endpoint="https://example/research")
    assert post.call_count == 1
    assert result["status"] == "success"
    # Endpoint override is respected.
    assert post.call_args.args[0] == "https://example/research"


def test_upload_cohort_skips_events_outside_cohort(tmp_db, monkeypatch):
    # Two events: one cohort-tagged, one not.
    monkeypatch.setenv("AGENTGUARD_RESEARCH_COHORT", "cohort-X")
    StatsCollector().record_tool_call("t_in", duration_ms=1)

    monkeypatch.delenv("AGENTGUARD_RESEARCH_COHORT")
    collector_mod._collector = None
    StatsCollector().record_tool_call("t_out", duration_ms=2)

    result = upload_cohort("cohort-X", dry_run=True)
    assert result["records_exported"] == 1
    tools = [e["tool_name"] for e in result["payload"]["events"]]
    assert tools == ["t_in"]
