---
doc_id: reader-security
surface: security-contract
owner: reader-maintainers
last_verified: 2026-07-28
summary: Trust boundaries, path safety, dependency posture, and safe defaults for Reader operations.
---

# Security

`reader` is a local experimental workbench, not a multi-tenant service. Its security model is therefore mostly about trust boundaries, safe defaults, and preventing accidental damage or ambiguous execution rather than defending an internet-facing API.

This document describes the current security posture and the boundaries maintainers should preserve.

## Trust Model

Assume these inputs may be malformed or mistaken:

- `config.yaml`
- resource paths
- raw input files
- CLI overrides
- generated manifests

Assume these components are trusted code and should be treated accordingly:

- built-in plugins
- external plugins discovered from Python entry points
- notebook templates
- repo-local Python code and dependencies

External plugins and notebooks are executable Python. They are an extension surface, not a sandbox boundary.

## Current Controls

### Config parsing

- YAML uses a SafeLoader-based parser that rejects duplicate mapping keys.
- Only `reader/v8` is accepted.
- removed config keys are rejected explicitly
- protocol, experiment, paths, resources, and annotations are shape-checked before model validation

### Schema strictness

- plugin configs use pydantic models with `extra = "forbid"`
- unsupported public keys fail fast
- CLI JSON modes reject unsupported combinations explicitly

### Filesystem safety

- `paths.<subdir>` must remain relative to `paths.outputs`
- path escapes via `..` are rejected
- absolute subdirectory paths are rejected
- generated runtime outputs are confined to each experiment's `outputs/`
- notebook artifact staging and publication reject path and symlink escapes

### Execution boundaries

- plugin ports are typed and validated
- dataframe contracts are checked at runtime
- built-in plugins are registered through an explicit manifest, not by implicit package scanning
- external plugins must come from the `reader.plugins` entry-point group

### Provenance and integrity

- schema-v5 records bind complete and effective producer-config digests plus
  Reader build identity
- data and file bundles are tracked in `outputs/manifests/records.json`
- direct and auto-discovered input files are confined to the experiment root;
  schema-v5 evidence records their relative path, byte size, SHA-256 digest,
  and selection policy
- dataframe and file-bundle artifacts carry byte sizes and SHA-256 digests that
  `reader verify` checks against current files
- input evidence is captured before execution and rechecked before catalog
  commit so a mid-run source change refuses persistence
- record readers accept schema v5 only; older payloads make the catalog invalid
  and require regeneration from the owning experiment

## What Reader Does Not Promise

`reader` does not currently provide:

- sandboxing for untrusted plugins
- sandboxing for notebook execution
- secret management
- network isolation for external code
- policy enforcement that only signed plugins may run

If you install or author a plugin, you are executing Python with the same trust level as the local environment.

## Maintainer Guidance

- Treat plugin additions as code-execution changes, not just config additions.
- Keep plugin implementations thin and push domain logic into `domains/`.
- Prefer explicit manifests and typed ports over implicit runtime discovery.
- Do not add config shims or raw graph mutation surfaces.
- Keep generated outputs generated; do not hand-edit manifests to “fix” runtime state.

## Operator Guidance

- Do not place secrets in `config.yaml`.
- Review external plugins before enabling them.
- Treat notebooks as executable code.
- Use `uv sync --locked` so dependency state matches the lockfile.
- Prefer preflight commands before mutation:
  `reader validate`, `reader explain`, `reader run --dry-run`.

## Security Review Checklist

Use this checklist for security-sensitive changes.

- Does the change widen the public config surface?
- Does it add a new path or file-binding surface?
- Does it introduce dynamic import or implicit plugin discovery?
- Does it weaken contract validation or port checks?
- Does it change where generated outputs or manifests are written?
- Does it add a new execution surface for notebooks or external code?

If the answer is yes to any of those, the change deserves an explicit security review.

## Current Open Security Debt

The main open risk is semantic drift. Reader limits it by compiling protocol
semantics and execution bindings into one inspectable program. Nodes that are
not executed must remain explicitly domain-defined or unavailable; they must
not acquire behavior through an unreported plugin branch.

The other standing boundary is external plugins. They remain a deliberate trust boundary rather than a sandboxed capability surface.

## Related Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [DESIGN.md](./DESIGN.md)
- [QUALITY.md](./QUALITY.md)
- [RELIABILITY.md](./RELIABILITY.md)
