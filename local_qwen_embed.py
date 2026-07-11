#!/usr/bin/env python3
"""Generate comparison embeddings with local Qwen3-VL-Embedding-8B weights."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

try:
    from .io_utils import (
        append_jsonl,
        completed_ids,
        load_manifest,
        make_result,
        normalize_vector,
    )
except ImportError:  # Direct execution: python tools/embedding_compat/local_qwen_embed.py
    from io_utils import (  # type: ignore[no-redef]
        append_jsonl,
        completed_ids,
        load_manifest,
        make_result,
        normalize_vector,
    )


DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_DIMENSION = 2048
DEFAULT_INSTRUCTION = "Retrieve video frames relevant to the user's query."


def build_local_inputs(records: list[dict[str, str]], instruction: str) -> list[dict[str, str]]:
    inputs: list[dict[str, str]] = []
    for record in records:
        model_input = (
            {"text": record["text"], "instruction": instruction}
            if "text" in record
            else {"image": record["image"], "instruction": instruction}
        )
        inputs.append(model_input)
    return inputs


def load_embedder_class(qwen_repo: str | None) -> Any:
    if qwen_repo:
        repo = Path(qwen_repo).expanduser().resolve()
        if not (repo / "src/models/qwen3_vl_embedding.py").is_file():
            raise RuntimeError(f"invalid Qwen3-VL-Embedding repository: {repo}")
        sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("src.models.qwen3_vl_embedding")
    except ImportError as exc:
        raise RuntimeError(
            "cannot import Qwen3VLEmbedder; pass --qwen-repo pointing to the cloned "
            "QwenLM/Qwen3-VL-Embedding repository"
        ) from exc
    return module.Qwen3VLEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local snapshot")
    parser.add_argument(
        "--qwen-repo",
        default=os.environ.get("QWEN3_VL_EMBEDDING_REPO", ""),
        help="Cloned QwenLM/Qwen3-VL-Embedding repository",
    )
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 64 <= args.dimension <= 4096:
        raise SystemExit("--dimension must be between 64 and 4096 for the local 8B model")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required in the local Qwen environment") from exc

    if not torch.cuda.is_available():
        print(
            "warning: CUDA is unavailable; Qwen3-VL-Embedding-8B will run on CPU and may be very slow",
            file=sys.stderr,
        )
    embedder_class = load_embedder_class(args.qwen_repo or None)
    dtype = getattr(torch, args.dtype)
    model = embedder_class(model_name_or_path=args.model, torch_dtype=dtype)

    records = load_manifest(args.manifest)
    if args.overwrite:
        Path(args.output).unlink(missing_ok=True)
    done = completed_ids(args.output)
    pending = [record for record in records if record["id"] not in done]
    if not pending:
        print(f"Nothing to do; {len(done)} records already completed.")
        return 0

    failures = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        try:
            raw = model.process(build_local_inputs(batch, args.instruction), normalize=False)
            raw_vectors = raw.detach().float().cpu().tolist()
            if len(raw_vectors) != len(batch):
                raise RuntimeError(
                    f"model returned {len(raw_vectors)} vectors for {len(batch)} inputs"
                )
            for record, raw_vector in zip(batch, raw_vectors):
                vector = normalize_vector(raw_vector, args.dimension)
                row = make_result(
                    record=record,
                    model=args.model,
                    dimension=args.dimension,
                    instruction=args.instruction,
                    embedding=vector,
                )
                row["status"] = "ok"
                append_jsonl(args.output, row)
                print(f"[{start + 1}/{len(pending)}] ok    {record['id']}")
        except Exception as exc:  # Preserve completed batches and make a rerun possible.
            failures += len(batch)
            for record in batch:
                append_jsonl(args.output, {
                    "schema_version": 1,
                    "id": record["id"],
                    "modality": "text" if "text" in record else "image",
                    "model": args.model,
                    "dimension": args.dimension,
                    "instruction": args.instruction,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"error {record['id']}: {exc}", file=sys.stderr)
    print(f"Completed {len(pending) - failures}/{len(pending)} records; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
