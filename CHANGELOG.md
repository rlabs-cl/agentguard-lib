# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
