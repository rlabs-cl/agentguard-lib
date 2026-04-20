# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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



