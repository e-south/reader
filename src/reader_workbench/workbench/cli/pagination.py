from __future__ import annotations

import base64
import hashlib
import json
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

DEFAULT_PAGE_LIMIT = 25
MAX_PAGE_LIMIT = 100
_TOKEN_SCHEMA = "reader.page/v1"
_MAX_TOKEN_LENGTH = 4096


class PageRequestError(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        super().__init__(reason)
        self.field = field


@dataclass(frozen=True)
class Page[T]:
    items: tuple[T, ...]
    limit: int
    truncated: bool
    continuation: str | None


def _selection_digest(selection: Mapping[str, object]) -> str:
    encoded = json.dumps(selection, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _encode_token(*, surface: str, selection_digest: str, after: str) -> str:
    payload = {
        "after": after,
        "schema": _TOKEN_SCHEMA,
        "selection": selection_digest,
        "surface": surface,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _decode_token(token: str) -> dict[str, str]:
    if not token or len(token) > _MAX_TOKEN_LENGTH:
        raise PageRequestError("continuation", "continuation must be a nonempty Reader page token")
    try:
        padding = "=" * (-len(token) % 4)
        raw = base64.b64decode(token + padding, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise PageRequestError("continuation", "continuation is not a valid Reader page token") from exc
    expected_keys = {"after", "schema", "selection", "surface"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise PageRequestError("continuation", "continuation is not a valid Reader page token")
    if any(not isinstance(payload[key], str) or not payload[key] for key in expected_keys):
        raise PageRequestError("continuation", "continuation is not a valid Reader page token")
    if payload["schema"] != _TOKEN_SCHEMA:
        raise PageRequestError("continuation", "continuation uses an unsupported Reader page-token schema")
    return payload


def _normalize_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_PAGE_LIMIT
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_LIMIT:
        raise PageRequestError("limit", f"limit must be between 1 and {MAX_PAGE_LIMIT}")
    return limit


def page_collection[T](
    items: Sequence[T],
    *,
    key: Callable[[T], str],
    surface: str,
    selection: Mapping[str, object],
    limit: int | None,
    continuation: str | None,
) -> Page[T]:
    """Return one deterministic keyset page bound to a semantic selection."""

    effective_limit = _normalize_limit(limit)
    keyed = sorted(((key(item), item) for item in items), key=lambda pair: pair[0])
    keys = [item_key for item_key, _ in keyed]
    if any(not item_key for item_key in keys):
        raise ValueError(f"{surface} collection contains an empty pagination key")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{surface} collection contains duplicate pagination keys")

    fingerprint = _selection_digest(selection)
    start = 0
    if continuation is not None:
        token = _decode_token(continuation)
        if token["surface"] != surface or token["selection"] != fingerprint:
            raise PageRequestError(
                "continuation",
                "continuation does not match this collection and filter selection",
            )
        start = bisect_right(keys, token["after"])

    selected = keyed[start : start + effective_limit]
    truncated = start + effective_limit < len(keyed)
    next_token = None
    if truncated and selected:
        next_token = _encode_token(
            surface=surface,
            selection_digest=fingerprint,
            after=selected[-1][0],
        )
    return Page(
        items=tuple(item for _, item in selected),
        limit=effective_limit,
        truncated=truncated,
        continuation=next_token,
    )
