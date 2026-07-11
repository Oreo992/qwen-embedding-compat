"""Shared JSONL contract for local and Model Studio embedding runners."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp": "image/bmp",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest = Path(path).expanduser().resolve()
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_number}: record must be an object")
        record_id = raw.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"line {line_number}: id must be a non-empty string")
        record_id = record_id.strip()
        if record_id in seen:
            raise ValueError(f"line {line_number}: duplicate id {record_id!r}")
        has_text = isinstance(raw.get("text"), str) and bool(raw["text"].strip())
        has_image = isinstance(raw.get("image"), str) and bool(raw["image"].strip())
        if has_text == has_image:
            raise ValueError(f"line {line_number}: provide exactly one non-empty text or image")
        if has_text:
            record = {"id": record_id, "text": raw["text"].strip()}
        else:
            image = Path(raw["image"]).expanduser()
            if not image.is_absolute():
                image = manifest.parent / image
            image = image.resolve()
            if not image.is_file():
                raise ValueError(f"line {line_number}: image not found: {image}")
            if image.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                raise ValueError(f"line {line_number}: unsupported image type: {image.suffix}")
            record = {"id": record_id, "image": str(image)}
        seen.add(record_id)
        records.append(record)
    if not records:
        raise ValueError("manifest contains no records")
    return records


def image_to_data_uri(path: str | Path) -> str:
    image = Path(path)
    mime = SUPPORTED_IMAGE_SUFFIXES.get(image.suffix.lower())
    if not mime:
        raise ValueError(f"unsupported image type: {image.suffix}")
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def normalize_vector(values: Iterable[float], dimension: int) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) < dimension:
        raise ValueError(f"embedding has {len(vector)} values, expected at least {dimension}")
    vector = vector[:dimension]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("embedding has zero L2 norm")
    return [value / norm for value in vector]


def make_result(
    *,
    record: dict[str, str],
    model: str,
    dimension: int,
    instruction: str,
    embedding: Iterable[float],
) -> dict[str, Any]:
    vector = list(embedding)
    if len(vector) != dimension:
        raise ValueError(f"embedding has {len(vector)} values, expected {dimension}")
    return {
        "schema_version": 1,
        "id": record["id"],
        "modality": "text" if "text" in record else "image",
        "model": model,
        "dimension": dimension,
        "instruction": instruction,
        "embedding": vector,
    }


def completed_ids(output_path: str | Path) -> set[str]:
    path = Path(output_path)
    if not path.exists():
        return set()
    done: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("status", "ok") == "ok" and isinstance(row.get("id"), str):
            done.add(row["id"])
    return done


def append_jsonl(output_path: str | Path, row: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
