"""
视频片段搜索服务：文本 → Qwen3-VL-Embedding → ChromaDB 检索 → 返回匹配片段
"""

import sys
import time
import base64
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# ─── 配置 ─────────────────────────────────────────────────────
MODEL_PATH = r"D:\qwen3-vl-embedding\models\Qwen3-VL-Embedding-8B"
CHROMA_DB_PATH = r"D:\qwen3-vl-embedding\data\chromadb"
THUMBNAILS_PATH = r"D:\qwen3-vl-embedding\data\thumbnails"
COLLECTION_NAME = "video_segments"
EMBEDDING_DIM = 2048
QUERY_INSTRUCTION = "Retrieve video frames relevant to the user's query."

# ─── 模型加载（必须在 chromadb 之前） ──────────────────────────
print("Loading Qwen3-VL-Embedding-8B model...")
t0 = time.time()
sys.path.insert(0, r"D:\qwen3-vl-embedding")
from qwen3_vl_embedding import Qwen3VLEmbedder
model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH, torch_dtype=torch.bfloat16)
print(f"Model loaded in {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# ─── ChromaDB（模型加载后才能 import） ────────────────────────
import chromadb
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
print(f"ChromaDB loaded: {collection.count()} segments")

# ─── FastAPI ──────────────────────────────────────────────────
app = FastAPI(title="Video Segment Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def embed_text(text: str) -> list[float]:
    inputs = [{"text": text, "instruction": QUERY_INSTRUCTION}]
    raw = model.process(inputs, normalize=False)
    vec = raw[0].detach().float().cpu().numpy()
    vec = vec[:EMBEDDING_DIM]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class SearchResult(BaseModel):
    segment_id: str
    video_name: str
    start_time: float
    end_time: float
    score: float
    thumbnail_url: str
    source_folder: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    elapsed_ms: int


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), top_k: int = Query(12, ge=1, le=50)):
    t0 = time.time()
    query_vec = embed_text(q)
    raw = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
    )
    results = []
    if raw["ids"] and raw["ids"][0]:
        for i, seg_id in enumerate(raw["ids"][0]):
            meta = raw["metadatas"][0][i]
            dist = raw["distances"][0][i]
            results.append(SearchResult(
                segment_id=seg_id,
                video_name=meta.get("video_name", ""),
                start_time=meta.get("start_time", 0),
                end_time=meta.get("end_time", 0),
                score=round(1 - dist, 4),
                thumbnail_url=f"/api/thumbnail/{Path(meta.get('keyframe_path', '')).name}",
                source_folder=meta.get("source_folder", ""),
            ))
    elapsed = int((time.time() - t0) * 1000)
    return SearchResponse(query=q, results=results, elapsed_ms=elapsed)


@app.get("/api/thumbnail/{filename}")
def get_thumbnail(filename: str):
    path = Path(THUMBNAILS_PATH) / filename
    if path.exists():
        return FileResponse(path, media_type="image/jpeg")
    return {"error": "not found"}


@app.get("/api/stats")
def stats():
    return {"total_segments": collection.count()}


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(r"D:\qwen3-vl-embedding\search.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
