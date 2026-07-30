from __future__ import annotations

from pathlib import Path

from reader.api.notebooks import load_notebook_context
from reader.tests.support.configs import base_reader_config, write_config


def test_load_notebook_context_opens_nested_generated_notebook_through_public_experiment(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.yaml", base_reader_config(experiment_id="notebook_context"))
    notebook_path = tmp_path / "outputs" / "notebooks" / "EDA.py"
    notebook_path.parent.mkdir(parents=True)
    notebook_path.write_text("import marimo\n", encoding="utf-8")

    context = load_notebook_context(notebook_path)

    assert context.experiment.config_path == config_path.resolve()
    assert context.experiment.identity.id == "notebook_context"
    assert context.experiment_root == tmp_path.resolve()
    assert context.outputs_dir == (tmp_path / "outputs").resolve()
    assert context.notebooks_dir == notebook_path.parent.resolve()
    assert isinstance(context.pipeline_step_ids, tuple)
