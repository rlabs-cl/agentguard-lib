"""Entry point for ``python -m agentguard``.

Historically this just launched the MCP server; it now also exposes a
small ``research`` subcommand for controlled-study participants. Any
other argument combination falls through to the MCP server, preserving
backwards compatibility with the prior invocation.

    python -m agentguard                       # start MCP server (default)
    python -m agentguard research upload       # upload cohort-tagged events
    python -m agentguard research upload --dry-run
    python -m agentguard research upload --cohort <id>
"""

from __future__ import annotations

import argparse
import json
import sys

from agentguard.mcp.server import run_mcp_server
from agentguard.stats.collector import current_research_cohort
from agentguard.stats.research import upload_cohort


def _cmd_research_upload(argv: list[str]) -> int:
    """Run ``research upload``. Returns process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m agentguard research upload",
        description="Upload cohort-tagged telemetry events to the research endpoint.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        help=(
            "Research cohort id. Defaults to the value of "
            "AGENTGUARD_RESEARCH_COHORT in the current environment."
        ),
    )
    parser.add_argument(
        "--endpoint",
        dest="endpoint",
        help="Override the research endpoint URL.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print the anonymised payload without sending.",
    )
    args = parser.parse_args(argv)

    cohort = args.cohort or current_research_cohort()
    if not cohort:
        print(
            "error: no cohort id provided. Pass --cohort <id> or "
            "export AGENTGUARD_RESEARCH_COHORT before uploading.",
            file=sys.stderr,
        )
        return 2

    result = upload_cohort(
        cohort_id=cohort,
        endpoint=args.endpoint,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(json.dumps(result["payload"], indent=2, default=str))
        return 0

    print(json.dumps(
        {k: v for k, v in result.items() if k != "payload"},
        indent=2,
        default=str,
    ))
    return 0 if result.get("status") == "success" else 1


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "research" and argv[1] == "upload":
        return _cmd_research_upload(argv[2:])
    # Default: MCP server. Preserves prior invocation contract.
    run_mcp_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
