"""进一步诊断：验证百炼API向量自身稳定性，以及本地模型信息"""

import json
import os
import sys
import numpy as np
import torch
import httpx

sys.path.insert(0, r"D:\qwen3-vl-embedding")
from qwen3_vl_embedding import Qwen3VLEmbedder

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"


def cos(a, b):
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def l2_normalize(vec):
    arr = np.array(vec, dtype=np.float64)
    return (arr / np.linalg.norm(arr)).tolist()


# 1. 检查本地模型配置
print("=" * 60)
print("1. 本地模型配置检查")
print("=" * 60)
config_path = r"D:\qwen3-vl-embedding\models\Qwen3-VL-Embedding-8B\config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
print(f"  model_type: {config.get('model_type')}")
print(f"  hidden_size: {config.get('hidden_size')}")
print(f"  num_hidden_layers: {config.get('num_hidden_layers')}")
print(f"  num_attention_heads: {config.get('num_attention_heads')}")
print(f"  vocab_size: {config.get('vocab_size')}")
print(f"  architectures: {config.get('architectures')}")

# 2. 加载模型
print("\n加载本地模型...")
model = Qwen3VLEmbedder(
    model_name_or_path=r"D:\qwen3-vl-embedding\models\Qwen3-VL-Embedding-8B",
    torch_dtype=torch.bfloat16,
)
print("模型加载完成")

# 3. 多次调用测试本地确定性
print("\n" + "=" * 60)
print("2. 本地模型确定性测试")
print("=" * 60)
text = "hello world"
inst = "Retrieve video frames relevant to the user's query."
vecs = []
for i in range(3):
    raw = model.process([{"text": text, "instruction": inst}], normalize=True)
    vec = raw[0].detach().float().cpu().tolist()
    vecs.append(vec)
    if i > 0:
        c = cos(vecs[0], vecs[i])
        print(f"  Run 0 vs Run {i}: cosine={c:.10f}")

# 4. API 稳定性测试
print("\n" + "=" * 60)
print("3. API 稳定性测试")
print("=" * 60)
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
client = httpx.Client(headers=headers, timeout=30)

api_vecs = []
for i in range(3):
    payload = {
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"text": text}]},
        "parameters": {"dimension": 2048, "instruct": inst},
    }
    resp = client.post(URL, json=payload)
    vec = resp.json()["output"]["embeddings"][0]["embedding"]
    api_vecs.append(vec)
    if i > 0:
        c = cos(api_vecs[0], api_vecs[i])
        print(f"  Run 0 vs Run {i}: cosine={c:.10f}")

# 5. 语义判别测试：两端各自是否能区分相似/不相似文本
print("\n" + "=" * 60)
print("4. 语义判别能力对比")
print("=" * 60)
queries = [
    ("A kitten on grass", "小猫在草地上"),
    ("A kitten on grass", "量子计算机原理"),
]
for a, b in queries:
    la = model.process([{"text": a, "instruction": inst}], normalize=True)[0].detach().float().cpu().tolist()
    lb = model.process([{"text": b, "instruction": inst}], normalize=True)[0].detach().float().cpu().tolist()
    local_sim = cos(la, lb)

    pa = {"model": "qwen3-vl-embedding", "input": {"contents": [{"text": a}]}, "parameters": {"dimension": 2048, "instruct": inst}}
    pb = {"model": "qwen3-vl-embedding", "input": {"contents": [{"text": b}]}, "parameters": {"dimension": 2048, "instruct": inst}}
    aa = client.post(URL, json=pa).json()["output"]["embeddings"][0]["embedding"]
    ab = client.post(URL, json=pb).json()["output"]["embeddings"][0]["embedding"]
    api_sim = cos(aa, ab)

    print(f"  '{a}' <-> '{b}'")
    print(f"    Local sim: {local_sim:.4f}  API sim: {api_sim:.4f}  diff: {abs(local_sim-api_sim):.4f}")

# 6. 各维度API向量与全维度向量的关系
print("\n" + "=" * 60)
print("5. API 不同维度向量一致性")
print("=" * 60)
api_dims = {}
for dim in [256, 512, 1024, 2048, 2560]:
    payload = {
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"text": text}]},
        "parameters": {"dimension": dim, "instruct": inst},
    }
    resp = client.post(URL, json=payload)
    api_dims[dim] = resp.json()["output"]["embeddings"][0]["embedding"]

base = api_dims[2560]
for dim in [256, 512, 1024, 2048]:
    truncated = l2_normalize(base[:dim])
    c = cos(truncated, api_dims[dim])
    print(f"  API 2560[:dim]_renorm vs API dim={dim}: cosine={c:.6f}")

# 7. 本地各维度一致性
print("\n" + "=" * 60)
print("6. 本地不同维度截取一致性")
print("=" * 60)
local_raw_full = model.process([{"text": text, "instruction": inst}], normalize=False)[0].detach().float().cpu().tolist()
for dim in [256, 512, 1024, 2048]:
    local_dim = l2_normalize(local_raw_full[:dim])
    api_dim = api_dims[dim]
    c = cos(local_dim, api_dim)
    print(f"  local_raw[:dim]_renorm vs API dim={dim}: cosine={c:.6f}")

client.close()
print("\nDone.")
