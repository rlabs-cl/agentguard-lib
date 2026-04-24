# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] — 2026-04-24

### Added
- **`confidentiality_policy` field on archetype schema** (`ConfidentialityPolicy` enum
  in `agentguard.archetypes.schema`). Four values: `transparent`, `attribution`,
  `paraphrase` (default), `proprietary`. Authors declare the policy in their archetype
  YAML; the MCP server injects a matching directive into every pipeline response
  (`skeleton`, `contracts_and_wiring`, `logic`, `validate`, `debug`, `migrate`).
- Built-in archetypes (`api_backend`, `cli_tool`, `debug_backend`, `debug_frontend`,
  `library`, `react_spa`, `script`, `software_architecture`, `web_app`) now declare
  `confidentiality_policy: transparent` — their criteria can be reproduced in full
  by downstream LLMs and reviewed openly by consuming users, as befits an OSS default.
- Public helper `agentguard.mcp.agent_tools._confidentiality_directive_for(policy)`
  resolving a policy value (enum or string) to its directive text. Unknown policies
  fall back safely to `paraphrase`.

### Changed
- **Default confidentiality behaviour is now `paraphrase`** when an archetype does not
  declare a policy explicitly. Previously the server injected the strictest directive
  unconditionally. The new default lets LLMs explain *what* an archetype checks in
  their own words, while still forbidding verbatim reproduction of criterion text.
  This is a user-facing change for consumers of third-party archetypes that do not
  yet declare a policy; authors who want the pre-0.15 strict behaviour should set
  `confidentiality_policy: proprietary` in their YAML.

### Backwards compatibility
- The historic constant `agentguard.mcp.agent_tools._CONFIDENTIALITY_DIRECTIVE` is
  preserved as an alias for the `proprietary` directive text. External code that
  imported it continues to compile and returns the strict directive; all first-party
  call sites now route through `_confidentiality_directive_for`.

### Rationale
- The previous blanket "must not reproduce, summarise, or paraphrase" directive was
  incompatible with OSS auditability: an end user has no way to inspect what quality
  gates their own self-hosted agent applies if the consuming LLM is forbidden from
  describing them at all. The tiered policy returns control to archetype authors:
  closed commercial archetypes can opt in to strict protection; open community
  archetypes can opt in to full transparency; the neutral middle ground (paraphrase)
  serves as sensible default that protects author IP without blinding users.

## [0.14.0] — 2026-04-20

### Added
- **Research cohort telemetry (opt-in).** When the user exports
  `AGENTGUARD_RESEARCH_COHORT=<id>`, every tool event recorded in the
  local stats DB is tagged with that cohort id. With the env var unset
  (the default), the column is `NULL` and the feature is invisible.
- `python -m agentguard research upload` CLI subcommand. Fetches
  cohort-tagged events, anonymises them (SHA256-based 12-char hashes
  for `project_path` and `project_name`; `parameters_summary` dropped
  entirely; `error_message` truncated to 200 chars), and POSTs to
  `https://api.agentguard.rlabs.cl/api/research/events`. Endpoint
  overridable via `AGENTGUARD_RESEARCH_ENDPOINT`. `--dry-run` prints
  the anonymised payload without sending.
- `agentguard.stats.research` module: `get_cohort_events`,
  `anonymise_events`, `upload_cohort` — publicly importable so research
  tooling can integrate without shelling out.

### Changed
- `tool_events` schema gains a `research_cohort_id TEXT` column,
  nullable, with an index for cohort-scoped queries. Existing
  databases are migrated idempotently via `PRAGMA table_info` +
  `ALTER TABLE` on the next `get_connection()` call; no manual step
  required.
- `StatsCollector.__init__` now snapshots the cohort id once at
  session start, so a mid-session `AGENTGUARD_RESEARCH_COHORT` flip
  does not split events across cohorts.

### Context
This release is the first piece of infrastructure for the controlled
studies described in the rlabs-lab paper series (papers 002 and 005).
It has no effect on commercial users who never set the env var — the
feature surface is zero bytes of telemetry, zero network calls, until
explicitly opted in.

## [0.13.0] — 2026-04-19

### Added
- Pipeline stages (`skeleton`, `contracts_and_wiring`, `logic`) now embed a
  `quality_contract` block at the top of their response. The block carries
  the archetype's `self_challenge_criteria`, expected directory/file shape,
  validation check list, and an enforcement note directing the generating
  agent to satisfy those criteria *during* generation — not only after, when
  `validate` runs. The contract is covered by the existing confidentiality
  directive; the agent may use it to guide output but must not expose it to
  the end user. Measured overhead is ~374 tokens per response on average
  across the 10 built-in archetypes (range: 188–759 tokens).

### Rationale
- Observed during internal dogfooding: long-document archetypes suffered
  drift across later sections because the rubric was out of attention while
  writing. Prior design forced a correction loop — writer produces, `validate`
  scores, writer rewrites. The new design makes prevention the primary
  lever and leaves `validate` as reinforcement of a decision already made.

### Notes
- Fully backward-compatible. Clients that ignore the new key see no change;
  clients that use it (agents generating code or docs) gain earlier signal.

## [0.12.3] — 2026-04-19

### Fixed
- urllib outbound calls now send an explicit `User-Agent` so Cloudflare's
  bot-protection layer in front of `api.agentguard.rlabs.cl` no longer
  returns 403 (code 1010) on valid API keys. Previously the MCP surfaced
  these as "API key is invalid or expired", which was misleading.
- `my_archetypes` distinguishes genuine auth failures from CDN challenges
  via body sniffing, so the error message matches the actual cause.

### Added
- `agentguard._http.make_request` helper that all internal HTTP call sites
  now route through, guaranteeing a consistent UA.

## [0.10.0] — 2026-04-07

### Added
- `skeleton` tool for generating file tree structures with responsibilities
- `contracts_and_wiring` combined tool for L2+L3 stubs with imports (saves ~15K tokens vs separate calls)
- `contracts` tool for L2 typed function/class stubs
- `wiring` tool for L3 import and call-chain connections
- `logic` tool for L4 function body implementation
- `digest` tool for compact project summaries
- `debug` tool for structured debugging protocol
- `migrate` tool for migration planning with compatibility checks
- `validate` tool for mechanical code checks (syntax, lint, types, structure)
- `list_archetypes` tool to list all available archetypes
- `get_archetype` tool for detailed archetype configuration
- `reload_archetypes` tool to pick up newly installed archetypes
- `trace_summary` tool for cost & token tracking
- `docs` tool for AgentGuard documentation lookup
- `update_agentguard` tool for version updates from PyPI
- Built-in archetypes: `api_backend`, `library`, `cli_tool`, `react_spa`, `web_app`, `script`, `debug_backend`, `debug_frontend`
- Support for Claude Desktop, Claude Code, Cursor, Windsurf, and direct Python usage
- Agent-native quality framework with structured guidance (no external LLM calls)
- Documentation and examples

### Changed
- Initial release — first stable version

[0.10.0]: https://github.com/rlabs-cl/agentguard-lib/releases/tag/v0.10.0



