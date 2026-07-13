"""Independent validation for BaseRender metadata carried by promoter bindings."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import PurePosixPath
from typing import Any

_IUPAC_DNA = re.compile(r"[ACGTRYSWKMBDHVN]+")
_DENSEGEN_KEYS = frozenset(
    {
        "part_kind",
        "role",
        "constraint_name",
        "sequence",
        "core_sequence",
        "variant_id",
        "spacer_length",
        "placement_index",
        "part_index",
        "regulator",
        "motif_id",
        "tfbs_id",
        "orientation",
        "offset",
        "offset_raw",
        "length",
        "end",
        "pad_left",
        "site_id",
    }
)
_GENBANK_KEYS = frozenset(
    {
        "feature_id",
        "feature_order",
        "feature_type",
        "label",
        "role_hint",
        "location_raw",
        "start_0",
        "end_0",
        "strand",
        "confidence",
    }
)


def densegen_annotations(
    value: object,
    *,
    required_regulators: object,
    sequence: str,
    alias: str,
) -> list[object]:
    """Return non-empty DenseGen annotations after contract and span checks."""

    annotations = _required_list(value, context=f"{alias}.DenseGen annotations")
    regulators = _required_list(required_regulators, context=f"{alias}.required regulators")
    regulator_ids = [_required_text(item, context=f"{alias}.required regulators") for item in regulators]
    if len(regulator_ids) != len(set(regulator_ids)):
        raise ValueError(f"{alias}.required regulators must be unique.")
    for index, raw in enumerate(annotations):
        item = _annotation_mapping(raw, allowed_keys=_DENSEGEN_KEYS, alias=alias, label="DenseGen", index=index)
        part_kind = _required_text(item.get("part_kind"), context=f"{alias}.DenseGen annotation {index}.part_kind")
        if part_kind not in {"tfbs", "fixed_element"}:
            raise ValueError(f"{alias}.DenseGen annotation {index}.part_kind is unsupported.")
        literal = _required_text(item.get("sequence"), context=f"{alias}.DenseGen annotation {index}.sequence")
        if literal != literal.upper() or _IUPAC_DNA.fullmatch(literal) is None:
            raise ValueError(f"{alias}.DenseGen annotation {index}.sequence must be uppercase IUPAC DNA.")
        if part_kind == "tfbs":
            start = _required_integer(item.get("offset"), context=f"{alias}.DenseGen annotation {index}.offset")
            length = _required_integer(item.get("length"), context=f"{alias}.DenseGen annotation {index}.length")
            end = _required_integer(item.get("end"), context=f"{alias}.DenseGen annotation {index}.end")
            if start < 0 or length < 1 or end != start + length or end > len(sequence) or len(literal) != length:
                raise ValueError(f"{alias}.DenseGen annotation {index} span is invalid for the canonical sequence.")
            orientation = _required_text(
                item.get("orientation"), context=f"{alias}.DenseGen annotation {index}.orientation"
            )
            if orientation not in {"fwd", "rev"}:
                raise ValueError(f"{alias}.DenseGen annotation {index}.orientation must be fwd or rev.")
            _required_text(item.get("regulator"), context=f"{alias}.DenseGen annotation {index}.regulator")
        else:
            role = _required_text(item.get("role"), context=f"{alias}.DenseGen annotation {index}.role")
            if role not in {"upstream", "downstream"}:
                raise ValueError(f"{alias}.DenseGen annotation {index}.role is unsupported.")
            _required_text(item.get("constraint_name"), context=f"{alias}.DenseGen annotation {index}.constraint_name")
    return annotations


def genbank_annotations(value: object, *, sequence: str, alias: str) -> list[object]:
    """Return non-empty GenBank annotations after exact-key and span checks."""

    annotations = _required_list(value, context=f"{alias}.GenBank annotations")
    for index, raw in enumerate(annotations):
        item = _annotation_mapping(raw, allowed_keys=_GENBANK_KEYS, alias=alias, label="GenBank", index=index)
        for field in ("feature_id", "feature_type", "label"):
            _required_text(item.get(field), context=f"{alias}.GenBank annotation {index}.{field}")
        start = _required_integer(item.get("start_0"), context=f"{alias}.GenBank annotation span {index}.start_0")
        end = _required_integer(item.get("end_0"), context=f"{alias}.GenBank annotation span {index}.end_0")
        if start < 0 or end <= start or end > len(sequence):
            raise ValueError(f"{alias}.GenBank annotation span {index} is invalid for the canonical sequence.")
        strand = _required_integer(item.get("strand"), context=f"{alias}.GenBank annotation {index}.strand")
        if strand not in {-1, 0, 1}:
            raise ValueError(f"{alias}.GenBank annotation {index}.strand must be -1, 0, or 1.")
    return annotations


def safe_relative_posix_reference(value: object, *, context: str) -> str:
    """Reject host-specific, expanding, absolute, or traversing artifact references."""

    text = _required_text(value, context=context)
    path = PurePosixPath(text)
    first = path.parts[0] if path.parts else ""
    if (
        "\\" in text
        or text.startswith("~")
        or path.is_absolute()
        or ".." in path.parts
        or ":" in first
        or text.startswith("//")
    ):
        raise ValueError(f"{context} must be a relative POSIX artifact reference.")
    return str(path)


def _required_list(value: object, *, context: str) -> list[object]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a sequence.")
    result = list(value)
    if not result:
        raise ValueError(f"{context} must be non-empty.")
    return result


def _annotation_mapping(
    value: object,
    *,
    allowed_keys: frozenset[str],
    alias: str,
    label: str,
    index: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{alias}.{label} annotation {index} must be a mapping.")
    extras = sorted(set(value) - allowed_keys)
    if extras:
        raise ValueError(f"{alias} has non-contract annotation fields: {extras}.")
    return value


def _required_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value.strip()


def _required_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} must be a finite integer.")
    return int(value)


__all__ = ["densegen_annotations", "genbank_annotations", "safe_relative_posix_reference"]
