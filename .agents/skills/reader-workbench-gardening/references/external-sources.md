# External sources

This repo-local skill is grounded first in repository sources:

- [ARCHITECTURE.md](../../../../ARCHITECTURE.md)
- [DESIGN.md](../../../../DESIGN.md)
- [QUALITY.md](../../../../QUALITY.md)
- [RELIABILITY.md](../../../../RELIABILITY.md)
- [docs/repo-change-gate.md](../../../../docs/repo-change-gate.md)

For the skill-development and harness pass on this skill, these official
sources informed the contract:

| URL | Retrieved | Mapped update |
| --- | --- | --- |
| https://learn.chatgpt.com/docs/build-skills | 2026-07-29 | Store shared repository skills under `.agents/skills`, keep each workflow focused, and use progressive disclosure. |
| https://developers.openai.com/plugins/build/skills | 2026-07-29 | Keep skills as workflow guidance; package them as plugins only for installable distribution or MCP-backed capabilities. |
| https://developers.openai.com/api/docs/guides/tools-connectors-mcp | 2026-07-29 | Reserve MCP for live external tools and context, with narrow tool exposure and explicit approval for sensitive actions. |
| https://openai.com/index/harness-engineering/ | 2026-04-19 | Add explicit endpoint contracts, deterministic validation, and feedback-loop framing rather than relying on generic architecture prose. |
| https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/ | 2026-04-19 | Keep the workflow incremental, tool-backed, and guardrail-driven, with clear instructions and bounded orchestration rather than jumping straight to a broad multi-agent pattern. |
| https://openai.com/business/guides-and-resources/how-openai-uses-codex/ | 2026-04-19 | Keep persistent repo context and agent instructions explicit, and improve the local environment with deterministic validation commands that reduce repeated errors. |

When future gardening cycles depend on behavior, standards, or tooling outside
this repository, add new rows with URL, retrieval date, and mapped update.
