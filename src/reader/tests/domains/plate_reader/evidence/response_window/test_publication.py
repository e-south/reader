from pathlib import Path

import pytest

from reader.domains.plate_reader.evidence.response_window.publication import bundle_publication


def test_bundle_publication_restores_previous_output_when_verification_fails(tmp_path: Path) -> None:
    destination = tmp_path / "evidence"
    destination.mkdir()
    (destination / "owner.txt").write_text("previous output", encoding="utf-8")

    with bundle_publication(
        destination,
        bundle_label="test-evidence",
        overwrite=True,
    ) as publication:
        (publication.staging / "owner.txt").write_text("replacement output", encoding="utf-8")

        def reject(_root: Path) -> None:
            raise RuntimeError("injected verification failure")

        with pytest.raises(RuntimeError, match="injected verification failure"):
            publication.publish(reject)

    assert (destination / "owner.txt").read_text(encoding="utf-8") == "previous output"
    assert list(tmp_path.glob(".evidence.staging-*")) == []
    assert list(tmp_path.glob(".evidence.backup-*")) == []
