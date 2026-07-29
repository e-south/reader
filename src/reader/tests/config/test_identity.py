from __future__ import annotations

from reader.workbench.config import ReaderSpec, reader_spec_digest


def _spec(*, title: str | None = None) -> ReaderSpec:
    return ReaderSpec.model_validate(
        {
            "schema": "reader/v8",
            "experiment": {"id": "example", "title": title},
            "protocol": {
                "id": "workbench/generic",
                "inputs": {},
                "analysis": {},
                "outputs": {},
            },
        }
    )


def test_reader_spec_digest_is_normalized_and_sensitive_to_semantics() -> None:
    implicit_defaults = _spec()
    explicit_defaults = ReaderSpec.model_validate(
        {
            "plotting": {"palette": "colorblind"},
            "paths": {
                "notebooks": "notebooks",
                "exports": "exports",
                "plots": "plots",
                "outputs": "./outputs",
            },
            "protocol": {
                "outputs": {"exports": {}, "plots": {}, "notebook": {}},
                "analysis": {},
                "inputs": {},
                "id": "workbench/generic",
            },
            "experiment": {"title": None, "lifecycle": "active", "id": "example"},
            "resources": {"by_id": {}},
            "annotations": {},
            "schema": "reader/v8",
        }
    )

    assert reader_spec_digest(implicit_defaults) == reader_spec_digest(explicit_defaults)
    assert reader_spec_digest(_spec(title="Changed")) != reader_spec_digest(implicit_defaults)
