from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from reader.contracts import builtin_contract_catalog
from reader.errors import ExecutionError, RecordError
from reader.protocols import ProtocolBinding, builtin_protocol_catalog
from reader.workbench import PluginSemantics
from reader.workbench.assets import build_plugin_asset
from reader.workbench.context import RunContext
from reader.workbench.engine import execute_step
from reader.workbench.engine.execution import _runtime_provenance_inputs
from reader.workbench.graph import FileRef, PluginStep, ProvenanceInput, RecordRef, ResourceRef
from reader.workbench.ports import dataframe_input, dataframe_output, file_path_input, file_path_output
from reader.workbench.records import RecordStore, verify_record_store
from reader.workbench.records.model import record_revision_digest
from reader.workbench.registry import Plugin, PluginConfig, Registry


def _frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": ["A1"],
            "time": [0.0],
            "channel": ["OD600"],
            "value": [value],
        }
    )


def _context(tmp_path: Path) -> RunContext:
    outputs = tmp_path / "outputs"
    return RunContext(
        exp_dir=tmp_path,
        outputs_dir=outputs,
        artifacts_dir=outputs / "artifacts",
        plots_dir=outputs / "plots",
        exports_dir=outputs / "exports",
        records_path=outputs / "manifests" / "records.json",
        logger=logging.getLogger("reader.tests.input_snapshot"),
        palette_book=None,
        protocol=builtin_protocol_catalog().bind(ProtocolBinding(id="workbench/generic")),
        config_digest="sha256:experiment-config",
    )


def _registry(plugin_id: str, plugin_cls: type[Plugin]) -> Registry:
    registry = Registry(contracts=builtin_contract_catalog())
    registry.register(
        build_plugin_asset(
            plugin_id=plugin_id,
            semantics=PluginSemantics(
                domain="generic",
                family="input_snapshot_test",
                summary="Deterministic mutable-input provenance probe.",
            ),
            plugin_cls=plugin_cls,
        )
    )
    return registry


def test_file_input_evidence_is_bound_before_plugin_mutates_source(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.txt"
    raw.parent.mkdir()
    raw.write_text("1", encoding="utf-8")

    class MutatingFilePlugin(Plugin):
        ConfigModel = PluginConfig

        @classmethod
        def input_ports(cls):
            return {"raw": file_path_input("raw")}

        @classmethod
        def output_ports(cls):
            return {"df": dataframe_output("df", "tidy.v1")}

        def run(self, ctx, inputs, cfg):
            del ctx, cfg
            consumed = float(inputs["raw"].read_text(encoding="utf-8"))
            inputs["raw"].write_text("2", encoding="utf-8")
            return {"df": _frame(consumed)}

    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    with pytest.raises(RecordError, match="changed after input evidence was captured"):
        execute_step(
            step=PluginStep(
                kind="pipeline",
                id="ingest",
                plugin="ingest/mutating_file",
                reads={"raw": FileRef(path=raw)},
            ),
            phase="pipeline",
            store=store,
            ctx=_context(tmp_path),
            registry=_registry("ingest/mutating_file", MutatingFilePlugin),
        )

    assert store.latest_dataframe("ingest/df") is None


def test_record_input_evidence_binds_the_revision_loaded_before_plugin_execution(tmp_path: Path) -> None:
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    first = store.persist_dataframe(
        producer_id="upstream",
        producer_plugin="ingest/synthetic",
        out_name="df",
        record_id="upstream/df",
        df=_frame(1.0),
        contract_id="tidy.v1",
        inputs=[],
        config_digest="sha256:experiment-config",
    )
    first_revision = record_revision_digest(first, outputs_dir=store.root)

    class MutatingRecordPlugin(Plugin):
        ConfigModel = PluginConfig

        @classmethod
        def input_ports(cls):
            return {"df": dataframe_input("df", "tidy.v1")}

        @classmethod
        def output_ports(cls):
            return {"df": dataframe_output("df", "tidy.v1")}

        def run(self, ctx, inputs, cfg):
            del ctx, cfg
            store.persist_dataframe(
                producer_id="upstream",
                producer_plugin="ingest/synthetic",
                out_name="df",
                record_id="upstream/df",
                df=_frame(2.0),
                contract_id="tidy.v1",
                inputs=[],
                config_digest="sha256:experiment-config",
            )
            return {"df": inputs["df"]}

    with pytest.raises(RecordError, match="changed after input evidence was captured"):
        execute_step(
            step=PluginStep(
                kind="pipeline",
                id="downstream",
                plugin="transform/mutating_record",
                reads={"df": RecordRef(record_id="upstream/df")},
            ),
            phase="pipeline",
            store=store,
            ctx=_context(tmp_path),
            registry=_registry("transform/mutating_record", MutatingRecordPlugin),
        )

    assert store.latest_dataframe("downstream/df") is None
    current_upstream = store.latest_dataframe("upstream/df")
    assert current_upstream is not None
    assert record_revision_digest(current_upstream, outputs_dir=store.root) != first_revision


def test_file_set_provenance_preserves_declared_resource_semantics(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.fcs"
    step = PluginStep(
        kind="pipeline",
        id="ingest",
        plugin="ingest/synthetic",
        reads={"raw": ResourceRef(resource_id="raw_fcs", path=raw)},
    )

    evidence = _runtime_provenance_inputs(step=step, inputs={"raw": (raw,)})

    assert evidence == [
        ProvenanceInput(
            label="raw[0]",
            ref=ResourceRef(resource_id="raw_fcs", path=raw),
            discovery_policy="declared_resource",
        )
    ]


def test_persistence_rejects_uncaptured_provenance_inputs(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.txt"
    raw.parent.mkdir()
    raw.write_text("1", encoding="utf-8")
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )

    with pytest.raises(RecordError, match="requires pre-captured RecordInputEvidence"):
        store.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synthetic",
            out_name="df",
            record_id="ingest/df",
            df=_frame(1.0),
            contract_id="tidy.v1",
            inputs=[ProvenanceInput(label="raw", ref=FileRef(path=raw))],
            config_digest="sha256:experiment-config",
        )

    assert store.latest_dataframe("ingest/df") is None


def test_persistence_rechecks_snapshot_before_catalog_commit(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "inputs" / "raw.txt"
    raw.parent.mkdir()
    raw.write_text("1", encoding="utf-8")
    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    captured = store.capture_inputs([ProvenanceInput(label="raw", ref=FileRef(path=raw))])
    original_to_parquet = pd.DataFrame.to_parquet

    def mutate_after_artifact_write(frame, path, *args, **kwargs):
        result = original_to_parquet(frame, path, *args, **kwargs)
        raw.write_text("2", encoding="utf-8")
        return result

    monkeypatch.setattr(pd.DataFrame, "to_parquet", mutate_after_artifact_write)

    with pytest.raises(RecordError, match="changed after input evidence was captured"):
        store.persist_dataframe(
            producer_id="ingest",
            producer_plugin="ingest/synthetic",
            out_name="df",
            record_id="ingest/df",
            df=_frame(1.0),
            contract_id="tidy.v1",
            inputs=captured,
            config_digest="sha256:experiment-config",
        )

    assert store.latest_dataframe("ingest/df") is None


def test_failed_file_output_rerun_restores_last_successful_bundle(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.txt"
    raw.parent.mkdir()
    raw.write_text("1", encoding="utf-8")

    class MutatingExportPlugin(Plugin):
        ConfigModel = PluginConfig

        @classmethod
        def input_ports(cls):
            return {"raw": file_path_input("raw")}

        @classmethod
        def output_ports(cls):
            return {"artifact": file_path_output("artifact")}

        def run(self, ctx, inputs, cfg):
            del cfg
            payload = inputs["raw"].read_text(encoding="utf-8")
            output = ctx.exports_dir / "report.txt"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            if payload == "2":
                inputs["raw"].write_text("3", encoding="utf-8")
            return {"artifact": output}

    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    step = PluginStep(
        kind="export",
        id="report",
        plugin="export/mutating_file",
        reads={"raw": FileRef(path=raw)},
    )
    registry = _registry("export/mutating_file", MutatingExportPlugin)
    context = _context(tmp_path)

    execute_step(step=step, phase="exports", store=store, ctx=context, registry=registry)
    first = store.read_record("export:report")
    output = context.exports_dir / "report.txt"
    assert output.read_text(encoding="utf-8") == "1"

    raw.write_text("2", encoding="utf-8")
    with pytest.raises(RecordError, match="changed after input evidence was captured"):
        execute_step(step=step, phase="exports", store=store, ctx=context, registry=registry)

    assert output.read_text(encoding="utf-8") == "1"
    assert store.read_record("export:report") == first
    raw.write_text("1", encoding="utf-8")
    verification = verify_record_store(
        store,
        experiment_root=tmp_path,
        expected_config_digest=context.config_digest,
    )
    assert verification["status"] == "ok"
    staging = context.outputs_dir / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_failed_file_output_catalog_write_restores_last_successful_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "inputs" / "raw.txt"
    raw.parent.mkdir()
    raw.write_text("1", encoding="utf-8")

    class ExportPlugin(Plugin):
        ConfigModel = PluginConfig

        @classmethod
        def input_ports(cls):
            return {"raw": file_path_input("raw")}

        @classmethod
        def output_ports(cls):
            return {"artifact": file_path_output("artifact")}

        def run(self, ctx, inputs, cfg):
            del cfg
            output = ctx.exports_dir / "report.txt"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(inputs["raw"].read_text(encoding="utf-8"), encoding="utf-8")
            return {"artifact": output}

    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    step = PluginStep(
        kind="export",
        id="report",
        plugin="export/file",
        reads={"raw": FileRef(path=raw)},
    )
    registry = _registry("export/file", ExportPlugin)
    context = _context(tmp_path)

    execute_step(step=step, phase="exports", store=store, ctx=context, registry=registry)
    first = store.read_record("export:report")
    output = context.exports_dir / "report.txt"

    raw.write_text("2", encoding="utf-8")

    def _fail_catalog(_catalog):
        raise RecordError("catalog failed")

    monkeypatch.setattr(store, "_write_catalog", _fail_catalog)
    with pytest.raises(RecordError, match="catalog failed"):
        execute_step(step=step, phase="exports", store=store, ctx=context, registry=registry)

    assert output.read_text(encoding="utf-8") == "1"
    assert store.read_record("export:report") == first
    staging = context.outputs_dir / ".staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_relative_file_output_is_promoted_from_staging_root(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.txt"
    raw.parent.mkdir()
    raw.write_text("payload", encoding="utf-8")

    class RelativeExportPlugin(Plugin):
        ConfigModel = PluginConfig

        @classmethod
        def input_ports(cls):
            return {"raw": file_path_input("raw")}

        @classmethod
        def output_ports(cls):
            return {"artifact": file_path_output("artifact")}

        def run(self, ctx, inputs, cfg):
            del cfg
            relative_path = Path("exports/report.txt")
            output = ctx.outputs_dir / relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(inputs["raw"].read_text(encoding="utf-8"), encoding="utf-8")
            return {"artifact": relative_path}

    store = RecordStore(
        tmp_path / "outputs",
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    execute_step(
        step=PluginStep(
            kind="export",
            id="report",
            plugin="export/relative",
            reads={"raw": FileRef(path=raw)},
        ),
        phase="exports",
        store=store,
        ctx=_context(tmp_path),
        registry=_registry("export/relative", RelativeExportPlugin),
    )

    output = tmp_path / "outputs" / "exports" / "report.txt"
    assert output.read_text(encoding="utf-8") == "payload"
    assert store.read_record("export:report").files == (output,)


def test_file_output_transaction_rejects_symlinked_staging_parent(tmp_path: Path) -> None:
    raw = tmp_path / "inputs" / "raw.txt"
    raw.parent.mkdir()
    raw.write_text("payload", encoding="utf-8")
    outputs = tmp_path / "outputs"
    store = RecordStore(
        outputs,
        contracts=builtin_contract_catalog(),
        experiment_root=tmp_path,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outputs / ".staging").symlink_to(outside, target_is_directory=True)
    calls: list[Path] = []

    class ExportPlugin(Plugin):
        ConfigModel = PluginConfig

        @classmethod
        def input_ports(cls):
            return {"raw": file_path_input("raw")}

        @classmethod
        def output_ports(cls):
            return {"artifact": file_path_output("artifact")}

        def run(self, ctx, inputs, cfg):
            del inputs, cfg
            calls.append(ctx.outputs_dir)
            return {"artifact": ctx.outputs_dir / "exports/report.txt"}

    with pytest.raises(ExecutionError, match="staging directory must stay within"):
        execute_step(
            step=PluginStep(
                kind="export",
                id="report",
                plugin="export/file",
                reads={"raw": FileRef(path=raw)},
            ),
            phase="exports",
            store=store,
            ctx=_context(tmp_path),
            registry=_registry("export/file", ExportPlugin),
        )

    assert calls == []
    assert list(outside.iterdir()) == []
