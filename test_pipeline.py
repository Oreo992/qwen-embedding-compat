"""快速测试：抽帧 + embedding 单个视频的完整流程"""

import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

VIDEO_DIR = Path(r"E:\美团视频")
MODEL_PATH = r"D:\qwen3-vl-embedding\models\Qwen3-VL-Embedding-8B"
THUMBNAILS = Path(r"D:\qwen3-vl-embedding\data\thumbnails")
THUMBNAILS.mkdir(parents=True, exist_ok=True)

# 找一个短视频
test_video = None
for f in sorted(VIDEO_DIR.rglob("*")):
    if f.is_file() and f.suffix.lower() == ".mp4":
        size_mb = f.stat().st_size / 1024 / 1024
        if 3 < size_mb < 10:
            test_video = f
            break

print(f"Test video: {test_video.name} ({test_video.stat().st_size/1024/1024:.1f} MB)")

# 1. 时长
container = av.open(str(test_video))
stream = container.streams.video[0]
if stream.duration and stream.time_base:
    duration = float(stream.duration * stream.time_base)
elif container.duration:
    duration = container.duration / av.time_base
else:
    duration = 0
container.close()
print(f"Duration: {duration:.1f}s")

# 2. 切片 (5s)
segments = []
t, idx = 0.0, 0
while t < duration:
    end = min(t + 5, duration)
    if end - t < 2 and idx > 0:
        break
    segments.append({"index": idx, "start": round(t, 2), "end": round(end, 2),
                     "target_time": round((t + end) / 2, 2)})
    t = end
    idx += 1

print(f"Segments: {len(segments)}")
for s in segments:
    print(f"  [{s['index']}] {s['start']:.2f}-{s['end']:.2f}s target={s['target_time']:.2f}s")

# 3. 抽帧（前 2 段）
frames = []
for seg in segments[:2]:
    container = av.open(str(test_video))
    stream = container.streams.video[0]
    target_pts = int(seg["target_time"] / stream.time_base)
    container.seek(target_pts, stream=stream)
    for f in container.decode(video=0):
        img = f.to_image()
        path = THUMBNAILS / f"test_seg{seg['index']:04d}.jpg"
        img.save(str(path), format="JPEG", quality=85)
        frames.append(str(path))
        print(f"  Frame [{seg['index']}] saved: {path.name} ({path.stat().st_size/1024:.1f}KB) size={img.size}")
        break
    container.close()

# 4. Embedding
print("\nLoading model...")
sys.path.insert(0, r"D:\qwen3-vl-embedding")
from qwen3_vl_embedding import Qwen3VLEmbedder
model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH, torch_dtype=torch.bfloat16)
print("Model loaded.")

instruction = "Retrieve video frames relevant to the user's query."
for i, fp in enumerate(frames):
    t0 = time.time()
    raw = model.process([{"image": fp, "instruction": instruction}], normalize=False)
    vec = raw[0].detach().float().cpu().numpy()
    vec_trunc = vec[:2048]
    norm = np.linalg.norm(vec_trunc)
    vec_trunc = vec_trunc / norm
    elapsed = time.time() - t0
    print(f"  Frame [{i}] raw_dim={len(vec)} trunc_dim={len(vec_trunc)} norm={np.linalg.norm(vec_trunc):.4f} time={elapsed:.2f}s")

# 5. ChromaDB 写入测试
import chromadb
chroma_path = r"D:\qwen3-vl-embedding\data\test_chromadb"
Path(chroma_path).mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=chroma_path)
col = client.get_or_create_collection(name="test_segments", metadata={"hnsw:space": "cosine"})

col.upsert(
    ids=["test_seg_0_5"],
    embeddings=[vec_trunc.tolist()],
    metadatas=[{
        "video_path": "",
        "video_name": test_video.name,
        "start_time": 0.0,
        "end_time": 5.0,
        "keyframe_path": frames[0],
        "duration": 5.0,
        "user_id": "test",
    }],
)
print(f"\nChromaDB test: count={col.count()}")

# 搜索测试
result = col.query(query_embeddings=[vec_trunc.tolist()], n_results=1)
print(f"Search result: {result['ids']}, distance={result['distances']}")

print("\nAll pipeline checks passed!")
