"""Small deterministic file-format helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=reject_duplicate_keys,
                      parse_constant=reject_constant)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_relative_file(base: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("file path must be a nonempty relative string")
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("artifact paths must remain inside their record directory")
    result = (base / p).resolve()
    if not result.is_relative_to(base.resolve()) or not result.is_file():
        raise ValueError(f"missing or out-of-directory artifact: {relative}")
    return result


def require_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"invalid SHA-256 record for {path.name}")
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")

