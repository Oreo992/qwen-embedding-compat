"""
视频批处理脚本：切片 → 抽帧 → 本地 Qwen3-VL-Embedding-8B → ChromaDB
与 videotest1 项目保持相同的 collection 结构和 segment_id 命名规则
"""

import io
import json
import math
import os
import re
import sys
import time
from functools import partial
from pathlib import Path

print = partial(print, flush=True)  # 确保实时输出

import av
import numpy as np
import torch
from PIL import Image

# chromadb 必须延迟 import，在模型加载到 GPU 之后才能 import
# 否则 chromadb 的 Rust 二进制和 onnxruntime DLL 会与 CUDA 冲突导致 crash
chromadb = None

# ─── 配置 ───────────────────────────────────────────────────────
VIDEO_DIR = Path(r"E:\美团视频")
MODEL_PATH = r"D:\qwen3-vl-embedding\models\Qwen3-VL-Embedding-8B"
CHROMA_DB_PATH = r"D:\qwen3-vl-embedding\data\chromadb"
THUMBNAILS_PATH = r"D:\qwen3-vl-embedding\data\thumbnails"
PROGRESS_FILE = r"D:\qwen3-vl-embedding\data\progress.json"

SEGMENT_DURATION = 5        # 每段 5 秒
MIN_TAIL_DURATION = 2       # 尾段不足 2 秒则丢弃
EMBEDDING_DIM = 2048        # Qwen3-VL-Embedding-8B 截取维度
MAX_IMAGE_DIM = 768         # 缩放到此尺寸以加速推理 (768px → ~1.2s/帧, 原始1280px → ~40s/帧)
JPEG_QUALITY = 85
COLLECTION_NAME = "video_segments"
USER_ID = "local_batch"     # 标记为批量处理的用户

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v", ".mkv", ".webm"}
INSTRUCTION = "Retrieve video frames relevant to the user's query."


# ─── ChromaDB ────────────────────────────────────────────────────
def init_chromadb():
    global chromadb
    import chromadb as _chromadb
    chromadb = _chromadb
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ─── 进度管理 ─────────────────────────────────────────────────────
def load_progress() -> dict:
    p = Path(PROGRESS_FILE)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_progress(progress: dict):
    p = Path(PROGRESS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── 视频处理 ─────────────────────────────────────────────────────
def safe_name(filename: str) -> str:
    """与 videotest1 相同的文件名清理规则"""
    stem = Path(filename).stem
    return re.sub(r'[^\w\-.]', '_', stem)


def build_segments(duration: float) -> list[dict]:
    """与 videotest1 前端 buildSegments 相同的切片逻辑"""
    segments = []
    t = 0.0
    idx = 0
    while t < duration:
        end = min(t + SEGMENT_DURATION, duration)
        if end - t < MIN_TAIL_DURATION and idx > 0:
            break
        segments.append({
            "index": idx,
            "start": round(t, 2),
            "end": round(end, 2),
            "target_time": round((t + end) / 2, 2),
        })
        t = end
        idx += 1
    return segments


def extract_frame(video_path: str, target_time: float) -> Image.Image | None:
    """用 PyAV 精确 seek 到指定时间点并抽取一帧"""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]

        target_pts = int(target_time / stream.time_base)
        container.seek(target_pts, stream=stream)

        for frame in container.decode(video=0):
            return frame.to_image()

        container.close()
    except Exception as e:
        print(f"  [WARN] 抽帧失败 time={target_time:.2f}s: {e}")
    return None


def resize_frame(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """缩放到 max_dim，保持比例"""
    if max(image.size) > max_dim:
        image = image.copy()
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return image


def frame_to_jpeg_bytes(image: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def get_video_duration(video_path: str) -> float | None:
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            dur = float(stream.duration * stream.time_base)
        elif container.duration:
            dur = container.duration / av.time_base
        else:
            dur = 0
        container.close()
        return dur
    except Exception as e:
        print(f"  [ERR] 无法读取时长: {e}")
        return None


# ─── Embedding ────────────────────────────────────────────────────
def load_model():
    print("=" * 60)
    print("加载 Qwen3-VL-Embedding-8B 模型...")
    t0 = time.time()
    sys.path.insert(0, r"D:\qwen3-vl-embedding")
    from qwen3_vl_embedding import Qwen3VLEmbedder
    model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH, torch_dtype=torch.bfloat16)
    elapsed = time.time() - t0
    print(f"模型加载完成，耗时 {elapsed:.1f}s")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU 显存占用: {mem:.2f} GB")
    return model


def embed_frame(model, image_path: str) -> list[float]:
    """对单张图片生成 embedding，截取到 EMBEDDING_DIM 维并归一化"""
    inputs = [{"image": image_path, "instruction": INSTRUCTION}]
    raw = model.process(inputs, normalize=False)
    vec = raw[0].detach().float().cpu().numpy()
    vec = vec[:EMBEDDING_DIM]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ─── 扫描视频文件 ─────────────────────────────────────────────────
def scan_videos() -> list[Path]:
    videos = []
    for f in sorted(VIDEO_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(f)
    return videos


# ─── 主流程 ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  视频批处理: 切片 → 抽帧 → Embedding → ChromaDB")
    print("=" * 60)

    # 扫描视频
    videos = scan_videos()
    print(f"\n发现 {len(videos)} 个视频文件")

    # 加载进度
    progress = load_progress()
    done_videos = set(progress.get("completed_videos", []))
    pending = [v for v in videos if str(v) not in done_videos]
    print(f"已完成: {len(done_videos)}, 待处理: {len(pending)}")

    if not pending:
        print("所有视频已处理完成！")
        return

    # 初始化（模型必须先于 chromadb 加载，避免 DLL 冲突）
    model = load_model()
    collection = init_chromadb()
    Path(THUMBNAILS_PATH).mkdir(parents=True, exist_ok=True)

    total_segments = 0
    total_errors = 0
    t_start = time.time()

    for vi, video_path in enumerate(pending):
        video_name = video_path.name
        sname = safe_name(video_name)
        rel_folder = video_path.parent.name

        print(f"\n{'─'*60}")
        print(f"[{vi+1}/{len(pending)}] {rel_folder}/{video_name}")

        # 获取时长
        duration = get_video_duration(str(video_path))
        if duration is None or duration < 1:
            print(f"  跳过: 无法读取时长或时长过短")
            total_errors += 1
            continue

        # 切片
        segments = build_segments(duration)
        print(f"  时长: {duration:.1f}s → {len(segments)} 个片段 (每段 {SEGMENT_DURATION}s)")

        seg_ok = 0
        seg_err = 0

        for seg in segments:
            seg_id = f"{sname}_{seg['start']:.0f}_{seg['end']:.0f}"
            kf_filename = f"{sname}_seg{seg['index']:04d}.jpg"
            kf_path = Path(THUMBNAILS_PATH) / kf_filename

            # 抽帧
            frame = extract_frame(str(video_path), seg["target_time"])
            if frame is None:
                print(f"  [{seg['index']}] 抽帧失败，跳过")
                seg_err += 1
                continue

            # 缩放 + 保存缩略图
            frame = resize_frame(frame)
            jpeg_bytes = frame_to_jpeg_bytes(frame)
            kf_path.write_bytes(jpeg_bytes)

            # 先保存临时图片给模型用，然后 embed
            try:
                embedding = embed_frame(model, str(kf_path))
            except Exception as e:
                print(f"  [{seg['index']}] embedding 失败: {e}")
                seg_err += 1
                continue

            # 存入 ChromaDB
            metadata = {
                "video_path": "",
                "video_name": video_name,
                "start_time": seg["start"],
                "end_time": seg["end"],
                "keyframe_path": str(kf_path),
                "duration": round(seg["end"] - seg["start"], 2),
                "user_id": USER_ID,
                "source_folder": rel_folder,
            }
            collection.upsert(
                ids=[seg_id],
                embeddings=[embedding],
                metadatas=[metadata],
            )
            seg_ok += 1
            total_segments += 1

            if seg_ok % 10 == 0 or seg["index"] == segments[-1]["index"]:
                print(f"  进度: {seg_ok}/{len(segments)} segments | "
                      f"总计: {total_segments} | "
                      f"耗时: {time.time()-t_start:.0f}s")

        if seg_err > 0:
            total_errors += seg_err
            print(f"  完成: {seg_ok} ok, {seg_err} errors")

        # 记录进度
        if "completed_videos" not in progress:
            progress["completed_videos"] = []
        progress["completed_videos"].append(str(video_path))
        progress["total_segments"] = total_segments
        progress["last_video"] = video_name
        save_progress(progress)

    # 汇总
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  处理完成!")
    print(f"{'='*60}")
    print(f"  处理视频数: {len(pending)}")
    print(f"  生成片段数: {total_segments}")
    print(f"  失败片段数: {total_errors}")
    print(f"  总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  ChromaDB: {CHROMA_DB_PATH}")
    print(f"  缩略图: {THUMBNAILS_PATH}")
    print(f"  Collection 总记录: {collection.count()}")
    if total_segments > 0:
        print(f"  平均每段耗时: {elapsed/total_segments:.1f}s")


if __name__ == "__main__":
    main()
