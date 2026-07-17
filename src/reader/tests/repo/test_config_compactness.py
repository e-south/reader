from __future__ import annotations

import pytest

from reader.tests.repo.config_inventory import RepoConfigEntry, repo_config_inventory

pytestmark = [pytest.mark.integration, pytest.mark.repo_matrix]

REPO_CONFIGS = repo_config_inventory()

DEFAULT_PATHS = {
    "outputs": "./outputs",
    "plots": "plots",
    "exports": "exports",
    "notebooks": "notebooks",
}


@pytest.mark.parametrize("entry", REPO_CONFIGS, ids=lambda entry: entry.rel)
def test_repo_experiment_configs_omit_redundant_defaults(entry: RepoConfigEntry) -> None:
    config_path = entry.path
    data = entry.data

    assert "semantics" not in data, (
        f"{config_path}: use annotations collections, orders, or ordered_state_spaces; not semantics"
    )
    assert "assay" not in data, f"{config_path}: use annotations, not assay"
    assert data.get("paths") != DEFAULT_PATHS, f"{config_path}: omit default paths block"
    assert data.get("plotting") != {"palette": "colorblind"}, f"{config_path}: omit default plotting.palette"

    experiment = data.get("experiment") or {}
    assert isinstance(experiment, dict), f"{config_path}: experiment block is required"
    assert experiment.get("id") == config_path.parent.name, (
        f"{config_path}: experiment.id should be explicit and match the experiment directory name"
    )
    if "title" in experiment and "id" in experiment:
        assert experiment["title"] != experiment["id"], f"{config_path}: omit experiment.title when it matches id"
    if "lifecycle" in experiment:
        assert experiment["lifecycle"] != "active", f"{config_path}: omit experiment.lifecycle when it is active"

    protocol = data.get("protocol") or {}
    assert isinstance(protocol, dict), f"{config_path}: protocol block is required"
    protocol_id = protocol.get("id")
    assert isinstance(protocol_id, str) and protocol_id.strip(), f"{config_path}: protocol.id must be explicit"

    inputs = protocol.get("inputs") or {}
    assert isinstance(inputs, dict), f"{config_path}: protocol.inputs must be a mapping when present"
    ingest = inputs.get("ingest") or {}
    fold_change = inputs.get("fold_change") or {}

    if protocol_id in {
        "plate_reader/dual_reporter_screen",
        "plate_reader/single_reporter_screen",
        "plate_reader/retron_sponge_screen",
        "logic/sfxi_screen",
    }:
        assert "channels" not in ingest, f"{config_path}: omit protocol.inputs.ingest.channels; the protocol owns them"

    if protocol_id in {
        "plate_reader/dual_reporter_screen",
        "plate_reader/single_reporter_screen",
        "plate_reader/retron_sponge_screen",
        "logic/sfxi_screen",
    }:
        assert "target" not in fold_change, (
            f"{config_path}: omit protocol.inputs.fold_change.target; the compiled assay ratio owns it"
        )

    if protocol_id == "plate_reader/dual_reporter_screen":
        analysis = protocol.get("analysis") or {}
        if isinstance(analysis, dict):
            crosstalk = analysis.get("crosstalk_pairs") or {}
            if isinstance(crosstalk, dict):
                assert "target" not in crosstalk, (
                    f"{config_path}: omit protocol.analysis.crosstalk_pairs.target; dual-reporter crosstalk is fixed "
                    "to YFP/CFP"
                )
