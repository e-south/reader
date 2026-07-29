from __future__ import annotations


def reader_command(*parts: object) -> str:
    tokens = ["uv run reader"]
    for part in parts:
        text = str(part).strip()
        if text:
            tokens.append(text)
    return " ".join(tokens)
