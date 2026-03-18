from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from reader.errors import RecordError
from reader.protocols import ProtocolBinding
from reader.runtime import ReaderRuntime, builtin_runtime
from reader.workbench.commands import reader_command
from reader.workbench.config import ReaderSpec
from reader.workbench.decl import WorkbenchDecl, build_workbench_decl
from reader.workbench.engine._shared import pipeline_has_plugin
from reader.workbench.experiments import discover_experiment_configs
from reader.workbench.graph import input_ref_to_dict, output_ref_to_dict, resolve_workbench
from reader.workbench.templates import resolve_notebook_template_descriptor


def load_job_models(job_path: Path, *, runtime: ReaderRuntime | None = None) -> tuple[ReaderSpec, WorkbenchDecl]:
    runtime = runtime or builtin_runtime()
    spec = ReaderSpec.load(job_path)
    return spec, build_workbench_decl(spec, source_path=job_path, protocols=runtime.protocols)


def has_sfxi_step(decl: WorkbenchDecl, *, runtime: ReaderRuntime) -> bool:
    return pipeline_has_plugin(decl, runtime=runtime, tag="sfxi")


def dataframe_record_contracts(
    outputs_dir: Path,
    *,
    runtime: ReaderRuntime,
    exact: str | None = None,
    prefix: str | None = None,
) -> list[str]:
    contract_catalog = runtime.contracts
    store = runtime.record_store(outputs_dir, create=False)
    if not store.catalog_exists():
        return []
    try:
        records = store.iter_latest_records(kind="dataframe_artifact")
    except RecordError:
        return []
    matches: list[str] = []
    for record in records:
        contract = record.contract_id
        if exact and contract_catalog.satisfies(actual=contract, expected=exact):
            matches.append(contract)
            continue
        if prefix and contract.startswith(prefix):
            matches.append(contract)
    return matches


def template_requirements_satisfied(
    template_name: str,
    decl: WorkbenchDecl,
    outputs_dir: Path,
    *,
    runtime: ReaderRuntime,
) -> bool:
    descriptor = resolve_notebook_template_descriptor(template_name)
    requirements = descriptor.capabilities.requires_any
    if not requirements:
        return True
    record_contracts: list[str] | None = None
    for requirement in requirements:
        if requirement.plugin and pipeline_has_plugin(decl, runtime=runtime, plugin=requirement.plugin):
            return True
        if requirement.domain and pipeline_has_plugin(decl, runtime=runtime, domain=requirement.domain):
            return True
        if requirement.tag and pipeline_has_plugin(decl, runtime=runtime, tag=requirement.tag):
            return True
        if requirement.record_contract or requirement.record_contract_prefix:
            if record_contracts is None:
                record_contracts = dataframe_record_contracts(
                    outputs_dir,
                    runtime=runtime,
                    exact=requirement.record_contract,
                    prefix=requirement.record_contract_prefix,
                )
            else:
                exact = requirement.record_contract
                prefix = requirement.record_contract_prefix
                record_contracts.extend(
                    dataframe_record_contracts(outputs_dir, runtime=runtime, exact=exact, prefix=prefix)
                )
            if record_contracts:
                return True
    return False


def bind_decl_protocol(*, decl: WorkbenchDecl, runtime: ReaderRuntime):
    return runtime.bind_protocol(decl.experiment_semantics.protocol)


def default_protocol_plan(*, descriptor, runtime: ReaderRuntime):
    bound_protocol = runtime.bind_protocol(ProtocolBinding(id=descriptor.protocol))
    return bound_protocol, bound_protocol.compile()


def default_notebook_name() -> str:
    return f"EDA_{datetime.now().strftime('%Y%m%d')}.py"


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def find_nearest_experiments_dir(start: Path) -> Path:
    for base in [start] + list(start.parents):
        candidate = base / "experiments"
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return (start / "experiments").resolve()


def find_jobs(root: Path, *, include_scaffolds: bool = False) -> list[Path]:
    return discover_experiment_configs(root, include_scaffolds=include_scaffolds)


def find_year_jobs(year: str, root: Path) -> list[Path]:
    year_str = str(year).strip()
    if not year_str:
        raise typer.BadParameter("--year cannot be empty")
    if not year_str.isdigit() or len(year_str) != 4:
        raise typer.BadParameter("--year expects a 4-digit year (e.g., 2025).")
    if not root.exists() or not root.is_dir():
        raise typer.BadParameter(f"Experiments root not found: {root}")
    year_dir = root / year_str
    if not year_dir.exists() or not year_dir.is_dir():
        raise typer.BadParameter(f"No experiments directory for year {year_str} under {root}.")
    jobs = find_jobs(year_dir)
    if not jobs:
        raise typer.BadParameter(f"No experiments found under {year_dir}.")
    return jobs


def infer_job_path(job: str | None) -> Path:
    if job:
        value = str(job).strip()
        path = Path(value)
        if path.exists():
            if path.is_dir():
                candidate = path / "config.yaml"
                if candidate.exists():
                    return candidate.resolve()
                raise typer.BadParameter(
                    f"CONFIG directory {path!s} has no 'config.yaml'. "
                    "Pass a file path, an experiment directory that contains config.yaml, or a numeric index "
                    f"(see '{reader_command('ls')}')."
                )
            return path.resolve()
        if value.isdigit():
            idx = int(value)
            root_path = find_nearest_experiments_dir(Path.cwd())
            jobs = find_jobs(root_path)
            if not jobs:
                raise typer.BadParameter(f"No experiments found under {root_path}. Use '{reader_command('ls')}' first.")
            if idx < 1 or idx > len(jobs):
                raise typer.BadParameter(
                    f"Experiment index out of range: {idx} (valid: 1..{len(jobs)} under {root_path}). "
                    f"Use '{reader_command('ls')}' to see the index numbers."
                )
            return jobs[idx - 1]
        raise typer.BadParameter(
            f"CONFIG not found: {job!r}. Pass a path to a config.yaml, an experiment directory, "
            f"or a numeric experiment index from '{reader_command('ls')}'."
        )

    cwd = Path.cwd()
    candidate = cwd / "config.yaml"
    if candidate.exists():
        return candidate.resolve()
    for base in cwd.parents:
        candidate = base / "config.yaml"
        if candidate.exists():
            return candidate.resolve()
    raise typer.BadParameter(
        "Missing CONFIG and no 'config.yaml' found in the current or parent directories. "
        "Run inside an experiment dir or pass a path to the config (or the experiment dir). "
        f"Tip: use '{reader_command('ls')}' to list experiments and pass its index."
    )


def format_job_arg(job: str | None) -> str | None:
    if job is None:
        return None
    value = str(job).strip()
    return value or None


def require_dataframe_records(decl: WorkbenchDecl, job_path: Path, *, runtime: ReaderRuntime) -> None:
    layout = decl.experiment_semantics.layout
    outputs_dir = layout.outputs_dir
    store = runtime.record_store(
        outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        create=False,
    )
    if not store.catalog_exists():
        raise RecordError(
            f"No outputs/manifests/records.json found. Run '{reader_command('run', job_path)}' first to generate dataframe records."
        )
    try:
        records = store.iter_latest_records(kind="dataframe_artifact")
    except RecordError as exc:
        raise RecordError(
            f"Could not read record catalog at {store.records_path}. Run '{reader_command('run', job_path)}' first."
        ) from exc
    if not records:
        raise RecordError(
            f"No dataframe records listed in outputs/manifests/records.json. Run '{reader_command('run', job_path)}' first."
        )


def append_journal(job_path: Path, command_line: str) -> None:
    exp_dir = job_path.parent
    journal = exp_dir / (
        "JOURNAL.md" if (exp_dir / "JOURNAL.md").exists() or not (exp_dir / "journal.md").exists() else "journal.md"
    )
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "" if journal.exists() else "# Experiment Journal\n\n"
    entry = f"### {ts}\n\n```\n{command_line}\n```\n\n"
    journal.write_text(
        header + (journal.read_text(encoding="utf-8") if journal.exists() else "") + entry,
        encoding="utf-8",
    )


def resolve_pipeline_step_id(decl: WorkbenchDecl, which: str) -> str:
    which_str = str(which).strip()
    pipeline = list(resolve_workbench(decl).pipeline)
    if any(step.id == which_str for step in pipeline):
        return which_str
    options = ", ".join(step.id for step in pipeline[:12])
    raise typer.BadParameter(
        f"Unknown pipeline step id '{which_str}'. Tip: use '{reader_command('steps')}' to list ids "
        f"(first few: {options}{' …' if len(pipeline) > 12 else ''})."
    )


def spec_to_dict(spec_obj) -> dict:
    if hasattr(spec_obj, "to_dict"):
        return spec_obj.to_dict()
    return {
        "id": spec_obj.id,
        "plugin": spec_obj.plugin,
        "reads": {key: input_ref_to_dict(value) for key, value in (spec_obj.reads or {}).items()},
        "with": dict(spec_obj.with_ or {}),
        "writes": {key: output_ref_to_dict(value) for key, value in (spec_obj.writes or {}).items()},
    }
