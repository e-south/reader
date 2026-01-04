import json

import pytest

from reader.core.artifacts import ArtifactStore
from reader.core.errors import ArtifactError


def test_manifest_invalid_json_raises(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "manifest.json").write_text("{not json", encoding="utf-8")
    store = ArtifactStore(outputs)
    with pytest.raises(ArtifactError):
        store.latest("df")


def test_manifest_missing_keys_raises(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "manifest.json").write_text(json.dumps({"artifacts": []}), encoding="utf-8")
    store = ArtifactStore(outputs)
    with pytest.raises(ArtifactError):
        store.latest("df")


def test_plots_manifest_invalid_raises(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "manifest.json").write_text(json.dumps({"artifacts": {}, "history": {}}), encoding="utf-8")
    (outputs / "plots_manifest.json").write_text(json.dumps({"plots": {}}), encoding="utf-8")
    store = ArtifactStore(outputs)
    with pytest.raises(ArtifactError):
        store.append_plot_entry({"step_id": "plot", "files": []})


def test_exports_manifest_invalid_raises(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "manifest.json").write_text(json.dumps({"artifacts": {}, "history": {}}), encoding="utf-8")
    (outputs / "exports_manifest.json").write_text(json.dumps({"exports": {}}), encoding="utf-8")
    store = ArtifactStore(outputs)
    with pytest.raises(ArtifactError):
        store.append_export_entry({"step_id": "export", "files": []})
