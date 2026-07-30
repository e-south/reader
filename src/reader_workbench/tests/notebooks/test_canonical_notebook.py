import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from reader_workbench.api import open_experiment
from reader_workbench.errors import ConfigError
from reader_workbench.runtime import builtin_runtime
from reader_workbench.tests.support.configs import base_reader_config, write_config
from reader_workbench.workbench.notebooks import CANONICAL_NOTEBOOK_ID
from reader_workbench.workbench.notebooks.scaffold import write_experiment_notebook


def _render_canonical_notebook(tmp_path: Path) -> tuple[Path, str]:
    rendered_path, changed = write_experiment_notebook(
        tmp_path / "EDA.py",
        experiment_root=tmp_path,
        notebooks_root=tmp_path,
        overwrite=True,
    )
    assert changed is True
    return rendered_path, rendered_path.read_text(encoding="utf-8")


def _is_app_cell(dec: ast.AST) -> bool:
    if isinstance(dec, ast.Call):
        dec = dec.func
    return (
        isinstance(dec, ast.Attribute)
        and isinstance(dec.value, ast.Name)
        and dec.value.id == "app"
        and dec.attr == "cell"
    )


class _Collector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # Comprehension targets are scoped to the comprehension in Python 3.
        self.visit(node.iter)
        for condition in node.ifs:
            self.visit(condition)


def _find_duplicates(template: str) -> set[str]:
    tree = ast.parse(template)
    seen: set[str] = set()
    dupes: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(_is_app_cell(dec) for dec in node.decorator_list):
            collector = _Collector()
            for stmt in node.body:
                collector.visit(stmt)
            for name in collector.names:
                if name.startswith("_"):
                    continue
                if name in seen:
                    dupes.add(name)
                seen.add(name)
    return dupes


def _undecorated_cell_lines(template: str) -> list[int]:
    tree = ast.parse(template)
    return [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_"
        and not any(_is_app_cell(decorator) for decorator in node.decorator_list)
    ]


def test_canonical_notebook_has_no_duplicate_globals(tmp_path: Path) -> None:
    _, body = _render_canonical_notebook(tmp_path)
    dupes = sorted(_find_duplicates(body))
    assert not dupes, f"{CANONICAL_NOTEBOOK_ID} defines duplicate non-private cell globals: {dupes}"


def test_canonical_notebook_does_not_leave_cell_functions_undecorated(tmp_path: Path) -> None:
    _, body = _render_canonical_notebook(tmp_path)
    lines = _undecorated_cell_lines(body)
    assert not lines, f"{CANONICAL_NOTEBOOK_ID} has cell functions without @app.cell at lines {lines}"


def test_canonical_notebook_parses(tmp_path: Path) -> None:
    _, body = _render_canonical_notebook(tmp_path)
    try:
        ast.parse(body)
    except SyntaxError as exc:  # pragma: no cover - explicit failure path
        raise AssertionError(f"{CANONICAL_NOTEBOOK_ID} has invalid syntax: {exc}") from exc


def test_canonical_notebook_does_not_advertise_omitted_notebooks_extra(tmp_path: Path) -> None:
    _, body = _render_canonical_notebook(tmp_path)
    assert "notebooks group" not in body
    assert "Install the `notebooks` extra." not in body


def test_canonical_notebook_renders_through_scaffold_and_passes_marimo_check(tmp_path: Path) -> None:
    run_marimo_check = importlib.util.find_spec("marimo") is not None
    rendered_path, content = _render_canonical_notebook(tmp_path)
    assert "__PLOT_SPECS__" not in content

    if not run_marimo_check:
        return
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "check", str(rendered_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Rendered notebooks failed marimo check:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.integration
def test_generated_generic_eda_executes_without_assay_identity_columns(tmp_path: Path) -> None:
    if importlib.util.find_spec("marimo") is None:
        pytest.skip("Marimo is unavailable; use a separately managed notebook environment to run execution tests.")

    config_path = write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="generic_eda_execution",
            title="Generic EDA execution",
        ),
    )
    runtime = builtin_runtime()
    experiment = open_experiment(config_path, runtime=runtime)
    declaration = experiment._declaration
    layout = declaration.experiment_semantics.layout
    store = runtime.record_store(
        layout.outputs_dir,
        plots_subdir=layout.plots_subdir,
        exports_subdir=layout.exports_subdir,
        experiment_root=declaration.experiment.root,
    )
    store.persist_dataframe(
        producer_id="measurements",
        producer_plugin="transform/example",
        out_name="df",
        record_id="measurements/df",
        df=pd.DataFrame(
            {
                "position": ["A1", "A1"],
                "time": [0.0, 1.0],
                "channel": ["signal", "signal"],
                "value": [1.0, 2.0],
                "sample_id": ["sample-1", "sample-1"],
            }
        ),
        contract_id="tidy.v1",
        inputs=[],
        config_digest=declaration.config_digest,
    )
    notebooks_root = layout.outputs_dir / layout.notebooks_subdir
    notebook_path, changed = write_experiment_notebook(
        notebooks_root / "EDA.py",
        experiment_root=declaration.experiment.root,
        notebooks_root=notebooks_root,
    )
    html_path = tmp_path / "eda.html"
    runtime_root = tmp_path / ".marimo-runtime"
    env = os.environ.copy()
    env.update(
        {
            "XDG_CONFIG_HOME": str(runtime_root / "config"),
            "XDG_STATE_HOME": str(runtime_root / "state"),
            "XDG_CACHE_HOME": str(runtime_root / "cache"),
            "MPLCONFIGDIR": str(runtime_root / "matplotlib"),
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(notebook_path),
            "--no-include-code",
            "--output",
            str(html_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert changed is True
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert html_path.is_file()
    assert "Generic EDA execution" in html_path.read_text(encoding="utf-8")


def test_notebook_writer_rejects_symlinked_notebooks_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    notebooks_root = tmp_path / "outputs" / "notebooks"
    notebooks_root.parent.mkdir()
    try:
        notebooks_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ConfigError, match="symlink path components"):
        write_experiment_notebook(
            notebooks_root / "review.py",
            experiment_root=tmp_path,
            notebooks_root=notebooks_root,
        )

    assert list(outside.iterdir()) == []


def test_notebook_writer_rejects_notebooks_root_outside_experiment(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    outside = tmp_path / "outside"

    with pytest.raises(ConfigError, match="must stay within"):
        write_experiment_notebook(
            outside / "review.py",
            experiment_root=experiment_root,
            notebooks_root=outside,
        )

    assert not outside.exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_notebook_writer_rejects_symlink_target_before_existing_file_handling(overwrite: bool, tmp_path: Path) -> None:
    notebooks_root = tmp_path / "outputs" / "notebooks"
    notebooks_root.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("original", encoding="utf-8")
    target = notebooks_root / "review.py"
    try:
        target.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ConfigError, match="must not be a symlink"):
        write_experiment_notebook(
            target,
            experiment_root=tmp_path,
            notebooks_root=notebooks_root,
            overwrite=overwrite,
        )

    assert outside.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("target_name", ["../escaped.py", "/tmp/escaped.py"])
def test_notebook_writer_rejects_target_outside_configured_root(target_name: str, tmp_path: Path) -> None:
    notebooks_root = tmp_path / "outputs" / "notebooks"
    target = notebooks_root / target_name

    with pytest.raises(ConfigError, match="configured notebooks root"):
        write_experiment_notebook(
            target,
            experiment_root=tmp_path,
            notebooks_root=notebooks_root,
        )


def test_notebook_writer_writes_normal_target_with_owned_context(tmp_path: Path) -> None:
    notebooks_root = tmp_path / "outputs" / "notebooks"
    target = notebooks_root / "review.py"

    rendered, changed = write_experiment_notebook(
        target,
        experiment_root=tmp_path,
        notebooks_root=notebooks_root,
    )

    assert changed is True
    assert rendered == target
    assert target.is_file()


def test_canonical_notebook_uses_only_the_verified_public_record_surface(tmp_path: Path) -> None:
    _, body = _render_canonical_notebook(tmp_path)
    assert "from reader_workbench.api import" in body
    assert "records(experiment)" in body
    assert "read_dataframe(" in body
    assert "row_limit=200" in body
    assert "reader_workbench.workbench.records" not in body
    assert "discover_dataframe_records" not in body
    assert "read_parquet" not in body
    assert "scan_event_table" not in body
    assert "__ALLOW_RECORD_SCAN__" not in body


def test_canonical_notebook_public_id_is_stable() -> None:
    assert CANONICAL_NOTEBOOK_ID == "notebook/eda"
