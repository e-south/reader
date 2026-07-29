"""Adversarial filesystem coverage for invocation-ledger publication."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from reader.errors import ExecutionError
from reader.workbench.engine.invocations import InvocationLedger
from reader.workbench.records.identity import BuildIdentity

_SOURCE_DIGEST = "sha256:" + ("a" * 64)


def test_invocation_ledger_rejects_a_hard_link_without_mutating_its_target(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    epoch_id = str(uuid4())
    ledger = InvocationLedger(
        experiment_root=tmp_path,
        outputs_dir=outputs,
        provenance_epoch_id=epoch_id,
    )
    ledger.path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    sentinel = b"outside evidence must remain byte-identical\n"
    outside.write_bytes(sentinel)
    try:
        os.link(outside, ledger.path)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    with pytest.raises(ExecutionError, match="single link"):
        ledger.append_attempt(
            config_digest="sha256:config",
            build_identity=BuildIdentity(reader_version="1.2.3", source_digest=_SOURCE_DIGEST),
            operation="run",
            selected_step_ids={"pipeline": [], "plots": [], "exports": []},
            declared_inputs=[],
        )

    assert outside.read_bytes() == sentinel
