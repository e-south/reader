from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reader.domains.promoter.candidate_bindings import (
    BINDING_SCHEMA_ID,
    load_promoter_candidate_bindings,
)
from reader.domains.promoter.candidate_bindings.loader import sha256_file


def test_load_candidate_bindings_accepts_one_exact_design_binding(tmp_path: Path) -> None:
    root = _write_binding_fixture(tmp_path)

    bindings = load_promoter_candidate_bindings(root)
    binding = bindings.resolve("spyP")

    assert bindings.schema_id == BINDING_SCHEMA_ID
    assert bindings.schema_version == "1"
    assert binding.reader_design_id == "spyP"
    assert binding.candidate_id == "candidate-spyp"
    assert binding.baserender_adapter_kind == "densegen_tfbs"
    assert binding.baserender_record == {
        "id": "candidate-spyp",
        "sequence": "ACGTACGT",
        "densegen__used_tfbs_detail": [_densegen_annotation()],
    }
    assert binding.binding_status == "resolved"
    assert binding.binding_method == "exact_alias"


def test_load_candidate_bindings_accepts_the_declared_manifest_resource(tmp_path: Path) -> None:
    root = _write_binding_fixture(tmp_path)

    bindings = load_promoter_candidate_bindings(root / "manifest.json")

    assert bindings.root == root.resolve()
    assert bindings.resolve("spyP").candidate_id == "candidate-spyp"


def test_reader_selects_only_reader_design_namespace(tmp_path: Path) -> None:
    root = _write_binding_fixture(tmp_path)
    frame = pd.read_parquet(root / "bindings.parquet")
    source_alias = frame.iloc[0].copy()
    source_alias["alias_namespace"] = "source.alias"
    source_alias["alias"] = "spyP"
    _rewrite_binding_table(root, pd.concat([frame, source_alias.to_frame().T], ignore_index=True))

    bindings = load_promoter_candidate_bindings(root)

    assert [row.reader_design_id for row in bindings.rows] == ["spyP"]


def test_reader_rejects_bundle_without_reader_design_namespace(tmp_path: Path) -> None:
    root = _write_binding_fixture(tmp_path)
    frame = pd.read_parquet(root / "bindings.parquet")
    frame.loc[0, "alias_namespace"] = "source.alias"
    _rewrite_binding_table(root, frame)

    with pytest.raises(ValueError, match="contain no 'reader.design_id' aliases"):
        load_promoter_candidate_bindings(root)


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("unknown_manifest_field", "manifest fields"),
        ("record_id", "record identity"),
        ("parquet_metadata", "Parquet metadata"),
        ("candidate_table", "candidate-table provenance"),
    ],
)
def test_load_candidate_bindings_rejects_contract_drift(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    root = _write_binding_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if drift == "unknown_manifest_field":
        manifest["legacy_aliases"] = []
    elif drift == "record_id":
        manifest["record"]["record_id"] = "resolved_designs"
    elif drift == "candidate_table":
        manifest["candidate_table"]["selection_sha256"] = "c" * 64
    else:
        table_path = root / "bindings.parquet"
        table = pq.read_table(table_path).replace_schema_metadata(
            {
                b"schema_id": BINDING_SCHEMA_ID.encode(),
                b"schema_version": b"999",
                b"study_id": b"stress_ethanol_cipro_growth",
                b"record_id": b"promoter_candidate_bindings/bindings",
            }
        )
        pq.write_table(table, table_path)
        manifest["record"]["sha256"] = sha256_file(table_path).removeprefix("sha256:")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_promoter_candidate_bindings(root)


def test_load_candidate_bindings_rejects_artifact_digest_drift(tmp_path: Path) -> None:
    root = _write_binding_fixture(tmp_path)
    path = root / "bindings.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="table digest mismatch"):
        load_promoter_candidate_bindings(root)


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("duplicate_alias", "typed aliases must be unique"),
        ("sequence_digest", "sequence digest mismatch"),
        ("raw_x", "columns must be exactly"),
        ("nonresolved_status", "exact resolved"),
        ("fuzzy_resolution", "exact resolved"),
        ("unsupported_adapter", "unsupported BaseRender adapter"),
        ("densegen_regulators", "required regulators"),
    ],
)
def test_load_candidate_bindings_rejects_identity_and_projection_drift(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    root = _write_binding_fixture(tmp_path)
    frame = pd.read_parquet(root / "bindings.parquet")
    if drift == "duplicate_alias":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif drift == "sequence_digest":
        frame.loc[0, "canonical_sequence"] = "ACGTACGA"
    elif drift == "raw_x":
        frame["latentdna__raw_x"] = [[0.1, 0.2]]
    elif drift == "nonresolved_status":
        frame.loc[0, "binding_status"] = "ambiguous"
    elif drift == "fuzzy_resolution":
        frame.loc[0, "binding_method"] = "prefix_alias"
    elif drift == "densegen_regulators":
        frame.at[0, "densegen__required_regulators"] = "not-a-list"
    else:
        frame.loc[0, "baserender_adapter_kind"] = "legacy_sequence_plot"
    _rewrite_binding_table(root, frame)

    with pytest.raises(ValueError, match=message):
        load_promoter_candidate_bindings(root)


def test_candidate_binding_resolution_rejects_an_unknown_reader_alias(tmp_path: Path) -> None:
    bindings = load_promoter_candidate_bindings(_write_binding_fixture(tmp_path))

    with pytest.raises(ValueError, match="matches=0"):
        bindings.resolve("not-a-reader-design")


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("densegen_annotations_empty", "DenseGen annotations"),
        ("densegen_regulators_empty", "required regulators"),
        ("densegen_regulator_blank", "required regulators"),
        ("densegen_annotation_nonmapping", "DenseGen annotation.*mapping"),
        ("densegen_annotation_extra", "non-contract annotation fields"),
        ("densegen_annotation_blank", "DenseGen annotation.*regulator"),
        ("genbank_features_empty", "GenBank annotations"),
        ("genbank_feature_nonmapping", "GenBank annotation.*mapping"),
        ("genbank_feature_extra", "non-contract annotation fields"),
        ("genbank_span_negative", "GenBank annotation span"),
        ("genbank_span_nonfinite", "GenBank annotation span"),
        ("genbank_span_reversed", "GenBank annotation span"),
        ("genbank_span_out_of_bounds", "GenBank annotation span"),
        ("genbank_label_blank", "GenBank annotation.*label"),
    ],
)
def test_candidate_bindings_reject_malformed_adapter_metadata(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    root = _write_binding_fixture(tmp_path)
    frame = pd.read_parquet(root / "bindings.parquet")
    if drift.startswith("genbank_"):
        _configure_genbank(frame)
    if drift == "densegen_annotations_empty":
        frame.at[0, "densegen__used_tfbs_detail"] = []
    elif drift == "densegen_regulators_empty":
        frame.at[0, "densegen__required_regulators"] = []
    elif drift == "densegen_regulator_blank":
        frame.at[0, "densegen__required_regulators"] = [""]
    elif drift == "densegen_annotation_nonmapping":
        frame.at[0, "densegen__used_tfbs_detail"] = ["not-a-mapping"]
    elif drift == "densegen_annotation_extra":
        frame.at[0, "densegen__used_tfbs_detail"] = [{**_densegen_annotation(), "source": "/host/path"}]
    elif drift == "densegen_annotation_blank":
        frame.at[0, "densegen__used_tfbs_detail"] = [{**_densegen_annotation(), "regulator": ""}]
    elif drift == "genbank_features_empty":
        frame.at[0, "seq_annot__features"] = []
    elif drift == "genbank_feature_nonmapping":
        frame.at[0, "seq_annot__features"] = ["not-a-mapping"]
    elif drift == "genbank_feature_extra":
        frame.at[0, "seq_annot__features"] = [{**_genbank_feature(), "source": "/host/path"}]
    elif drift == "genbank_span_negative":
        frame.at[0, "seq_annot__features"] = [{**_genbank_feature(), "start_0": -1}]
    elif drift == "genbank_span_nonfinite":
        frame.at[0, "seq_annot__features"] = [{**_genbank_feature(), "end_0": float("inf")}]
    elif drift == "genbank_span_reversed":
        frame.at[0, "seq_annot__features"] = [{**_genbank_feature(), "start_0": 4, "end_0": 4}]
    elif drift == "genbank_span_out_of_bounds":
        frame.at[0, "seq_annot__features"] = [{**_genbank_feature(), "end_0": 9}]
    elif drift == "genbank_label_blank":
        frame.at[0, "seq_annot__features"] = [{**_genbank_feature(), "label": ""}]
    _rewrite_binding_table(root, frame)

    with pytest.raises(ValueError, match=message):
        load_promoter_candidate_bindings(root)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        r"nested\record.gb",
        r"C:\record.gb",
        r"\\server\share.gb",
        "~/record.gb",
        "C:record.gb",
        "/record.gb",
        "../record.gb",
    ],
)
def test_candidate_bindings_reject_unsafe_genbank_references(tmp_path: Path, unsafe_path: str) -> None:
    root = _write_binding_fixture(tmp_path)
    frame = pd.read_parquet(root / "bindings.parquet")
    _configure_genbank(frame)
    frame.loc[0, "seq_annot__source_file"] = unsafe_path
    _rewrite_binding_table(root, frame)

    with pytest.raises(ValueError, match="relative POSIX artifact reference"):
        load_promoter_candidate_bindings(root)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        r"nested\records.parquet",
        r"C:\records.parquet",
        r"\\server\records.parquet",
        "~/records.parquet",
        "C:records.parquet",
        "/records.parquet",
        "../records.parquet",
    ],
)
def test_candidate_bindings_reject_unsafe_manifest_source_references(tmp_path: Path, unsafe_path: str) -> None:
    root = _write_binding_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_artifacts"][0]["path"] = unsafe_path
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative POSIX artifact reference"):
        load_promoter_candidate_bindings(root)


def _write_binding_fixture(tmp_path: Path, *, reader_design_id: str = "spyP") -> Path:
    root = tmp_path / "bindings"
    root.mkdir()
    sequence = "ACGTACGT"
    sequence_digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    row = {
        "alias_namespace": "reader.design_id",
        "alias": reader_design_id,
        "display_label": f"{reader_design_id} promoter",
        "candidate_id": "candidate-spyp",
        "canonical_sequence": sequence,
        "sequence_sha256": sequence_digest,
        "candidate_table_id": "candidate-table",
        "candidate_selection_sha256": "b" * 64,
        "sequence_authority_dataset_id": "source-dataset",
        "sequence_authority_id": "source-row-1",
        "sequence_authority_sha256": "a" * 64,
        "source_class": "measured_reference",
        "design_family": "stress_promoter",
        "baserender_adapter_kind": "densegen_tfbs",
        "baserender_annotation_column": "densegen__used_tfbs_detail",
        "densegen__plan": "plan-v1",
        "densegen__run_id": "run-v1",
        "densegen__sampling_library_hash": "library-v1",
        "densegen__used_tfbs_detail": [_densegen_annotation()],
        "densegen__required_regulators": ["CpxR"],
        "seq_annot__features": None,
        "seq_annot__source_file": None,
        "usr_label__primary": None,
        "derived__product_kind": None,
        "binding_status": "resolved",
        "binding_method": "exact_alias",
    }
    table_path = root / "bindings.parquet"
    table = pa.Table.from_pandas(pd.DataFrame([row]), preserve_index=False).replace_schema_metadata(
        {
            b"schema_id": BINDING_SCHEMA_ID.encode(),
            b"schema_version": b"1",
            b"study_id": b"stress_ethanol_cipro_growth",
            b"record_id": b"promoter_candidate_bindings/bindings",
        }
    )
    pq.write_table(table, table_path)
    manifest = {
        "schema_id": BINDING_SCHEMA_ID,
        "schema_version": "1",
        "study_id": "stress_ethanol_cipro_growth",
        "created_at": "2026-07-13T00:00:00+00:00",
        "record": {
            "record_id": "promoter_candidate_bindings/bindings",
            "path": "bindings.parquet",
            "sha256": sha256_file(table_path).removeprefix("sha256:"),
            "row_count": 1,
        },
        "candidate_table": {"dataset_id": "candidate-table", "selection_sha256": "b" * 64},
        "source_artifacts": [{"artifact_id": "alias-authority", "path": "inputs/aliases.parquet", "sha256": "0" * 64}],
        "baserender_contract": {
            "contract_id": "dnadesign.baserender.sequence_panel.v1",
            "contract_version": "1",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _rewrite_binding_table(root: Path, frame: pd.DataFrame) -> None:
    table_path = root / "bindings.parquet"
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {
            b"schema_id": BINDING_SCHEMA_ID.encode(),
            b"schema_version": b"1",
            b"study_id": b"stress_ethanol_cipro_growth",
            b"record_id": b"promoter_candidate_bindings/bindings",
        }
    )
    pq.write_table(table, table_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record"]["sha256"] = sha256_file(table_path).removeprefix("sha256:")
    manifest["record"]["row_count"] = len(frame)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _densegen_annotation() -> dict[str, object]:
    return {
        "part_kind": "tfbs",
        "sequence": "ACGT",
        "regulator": "CpxR",
        "orientation": "fwd",
        "offset": 0,
        "length": 4,
        "end": 4,
    }


def _genbank_feature() -> dict[str, object]:
    return {
        "feature_id": "feature-1",
        "feature_order": 1,
        "feature_type": "misc_feature",
        "label": "promoter",
        "role_hint": "promoter",
        "location_raw": "[0:4](+)",
        "start_0": 0,
        "end_0": 4,
        "strand": 1,
        "confidence": "high",
    }


def _configure_genbank(frame: pd.DataFrame) -> None:
    frame.loc[0, "baserender_adapter_kind"] = "usr_genbank_annotations_v1"
    frame.loc[0, "baserender_annotation_column"] = "seq_annot__features"
    for field in (
        "densegen__plan",
        "densegen__run_id",
        "densegen__sampling_library_hash",
        "densegen__used_tfbs_detail",
        "densegen__required_regulators",
    ):
        frame.at[0, field] = None
    frame.at[0, "seq_annot__features"] = [_genbank_feature()]
    frame.loc[0, "seq_annot__source_file"] = "_artifacts/genbank/source.gb"
    frame.loc[0, "usr_label__primary"] = "spyP"
    frame.loc[0, "derived__product_kind"] = "promoter"
