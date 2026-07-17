# Gardening checklists

Use these lenses to keep the audit concrete and to avoid collapsing the task
into a vague "architecture review."

## Information ownership

- Does `reader/v8` stay the authored source of truth?
- Are assay-facing semantics owned by protocols rather than plugins?
- Is domain math or parsing living in `domains/` instead of CLI or plugin glue?
- Are docs pointing to canonical sources instead of forking duplicate guidance?

## Monolith pressure

- Are protocol families or compiler branches collecting too many
  responsibilities?
- Are notebook or report flows hiding domain semantics inside one large helper?
- Does one file or module own configuration, behavior, and rendering at once?
- Would a new assay family force edits across too many unrelated files?

## Assay lock-in

- Do naming, defaults, or CLI surfaces assume one assay family is the norm?
- Are new behaviors exposed semantically or through raw plugin-shaped config?
- Would adding a new assay require compatibility shims instead of new protocol
  ownership?

## Legacy creep and silent fallback

- Are removed legacy keys or behaviors trying to return through compatibility
  shims?
- Does any changed surface quietly coerce, infer, or ignore invalid states
  instead of failing fast?
- Do JSON or CLI surfaces hide empty or invalid selections as success?

## Docs and semantics drift

- Do docs still describe the current `reader/v8` surface?
- Are removed legacy keys or behaviors still documented or silently accepted?
- Does `AGENTS.md` route to the current maintainer workflow instead of stale
  instructions?

## Harness and CLI drift

- Are JSON surfaces deterministic and aligned with table surfaces?
- Can agents discover, inspect, validate, and dry-run without mutation?
- Do representative commands fail fast when invariants are violated?
- Are records and outputs still traceable through manifest-backed provenance?

## Directory and boundary drift

- Are generated outputs still treated as generated?
- Does code placement match the layer described in `ARCHITECTURE.md`?
- Are new helpers reducing coupling, or just moving the monolith around?

## Adjacent route boundaries

- Is this really new-experiment intake, metadata staging, or local experiment
  auditing instead of workbench gardening?
- Is the real task plugin implementation or protocol feature delivery rather
  than boundary hardening?
- Has the task expanded into branch, publish, or CI topology and therefore
  needs `docs/repo-maintenance.md`?

## Evidence capture

- Which canonical docs were checked first?
- Which code paths or modules support the claim?
- Which CLI probes confirm the runtime-facing statement?
- Which findings are verified fact versus inference or follow-up hypothesis?
