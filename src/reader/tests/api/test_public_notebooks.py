from __future__ import annotations

from pathlib import Path

import pytest

from reader.api.notebooks import load_notebook_context, resolve_effective_step_config
from reader.errors import ConfigError
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


def test_load_notebook_context_projects_generic_compiled_step_configs(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.yaml",
        base_reader_config(
            experiment_id="compiled_notebook_context",
            protocol_id="logic/sfxi_screen",
            protocol_inputs={"state_map_ref": "induction_logic"},
            resources={"sample_map": {"kind": "file", "path": "./inputs/metadata.csv"}},
            annotations={
                "ordered_state_spaces": {
                    "induction_logic": {
                        "column": "treatment",
                        "state_order": ["00", "10", "01", "11"],
                        "values": {
                            "00": "none",
                            "10": "a",
                            "01": "b",
                            "11": "a+b",
                        },
                        "case_sensitive": False,
                    }
                }
            },
        ),
    )

    context = load_notebook_context(config_path)

    step = next(item for item in context.pipeline_steps if item.plugin_id == "transform/sfxi")
    assert step.step_id == "sfxi_vec8"
    assert step.domain == "logic"
    assert step.family == "summary_transform"
    assert "sfxi" in step.tags

    resolved = resolve_effective_step_config(context.experiment, step.step_id)
    assert resolved.step == step
    assert resolved.values["state_map_ref"] == "induction_logic"
    assert resolved.values["response"] == {"logic_channel": "YFP/CFP", "intensity_channel": "YFP/OD600"}
    with pytest.raises(TypeError):
        resolved.values["state_map_ref"] = "changed"

    state_space = context.ordered_state_spaces["induction_logic"]
    assert state_space.column == "treatment"
    assert state_space.state_ids == ("00", "10", "01", "11")
    assert state_space.source_values == {"00": "none", "10": "a", "01": "b", "11": "a+b"}
    assert state_space.case_sensitive is False
    with pytest.raises(TypeError):
        context.ordered_state_spaces["other"] = state_space

    with pytest.raises(ConfigError, match="compiled pipeline step 'missing'"):
        resolve_effective_step_config(context.experiment, "missing")
