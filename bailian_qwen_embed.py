#!/usr/bin/env python3
"""Generate comparison embeddings with Alibaba Cloud Model Studio."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    from .io_utils import (
        append_jsonl,
        completed_ids,
        image_to_data_uri,
        load_manifest,
        make_result,
        normalize_vector,
    )
except ImportError:  # Direct execution: python tools/embedding_compat/bailian_qwen_embed.py
    from io_utils import (  # type: ignore[no-redef]
        append_jsonl,
        completed_ids,
        image_to_data_uri,
        load_manifest,
        make_result,
        normalize_vector,
    )


DEFAULT_MODEL = "qwen3-vl-embedding"
DEFAULT_DIMENSION = 2048
DEFAULT_INSTRUCTION = "Retrieve video frames relevant to the user's query."
DEFAULT_API_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def build_bailian_payload(
    record: dict[str, str], *, model: str, dimension: int, instruction: str
) -> dict[str, Any]:
    content = (
        {"text": record["text"]}
        if "text" in record
        else {"image": image_to_data_uri(record["image"])}
    )
    return {
        "model": model,
        "input": {"contents": [content]},
        "parameters": {
            "dimension": dimension,
            "instruct": instruction,
            "enable_fusion": False,
        },
    }


def extract_embedding(body: Any) -> list[float]:
    try:
        vector = body["output"]["embeddings"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Model Studio response does not contain an embedding vector") from exc
    if not isinstance(vector, list) or not vector:
        raise ValueError("Model Studio response does not contain an embedding vector")
    return [float(value) for value in vector]


def request_embedding(
    client: Any,
    record: dict[str, str],
    *,
    api_url: str,
    model: str,
    dimension: int,
    instruction: str,
    retries: int,
) -> dict[str, Any]:
    payload = build_bailian_payload(
        record, model=model, dimension=dimension, instruction=instruction
    )
    for attempt in range(retries + 1):
        response = client.post(api_url, json=payload)
        if response.status_code < 400:
            vector = normalize_vector(extract_embedding(response.json()), dimension)
            return make_result(
                record=record,
                model=model,
                dimension=dimension,
                instruction=instruction,
                embedding=vector,
            )
        if response.status_code not in RETRYABLE_STATUS or attempt == retries:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(f"Model Studio HTTP {response.status_code}: {detail}")
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dimension not in {256, 512, 768, 1024, 1536, 2048, 2560}:
        raise SystemExit("--dimension must be supported by Model Studio")
    if args.concurrency < 1 or args.retries < 0:
        raise SystemExit("--concurrency must be >= 1 and --retries must be >= 0")
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("httpx is required: python -m pip install httpx") from exc

    records = load_manifest(args.manifest)
    if args.overwrite:
        from pathlib import Path

        Path(args.output).unlink(missing_ok=True)
    done = completed_ids(args.output)
    pending = [record for record in records if record["id"] not in done]
    if not pending:
        print(f"Nothing to do; {len(done)} records already completed.")
        return 0

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    failures = 0
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    request_embedding,
                    client,
                    record,
                    api_url=args.api_url,
                    model=args.model,
                    dimension=args.dimension,
                    instruction=args.instruction,
                    retries=args.retries,
                ): record
                for record in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                record = futures[future]
                try:
                    row = future.result()
                    row["status"] = "ok"
                    print(f"[{index}/{len(pending)}] ok    {record['id']}")
                except Exception as exc:  # Preserve other successful records.
                    failures += 1
                    row = {
                        "schema_version": 1,
                        "id": record["id"],
                        "modality": "text" if "text" in record else "image",
                        "model": args.model,
                        "dimension": args.dimension,
                        "instruction": args.instruction,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[{index}/{len(pending)}] error {record['id']}: {exc}", file=sys.stderr)
                append_jsonl(args.output, row)
    print(f"Completed {len(pending) - failures}/{len(pending)} records; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
