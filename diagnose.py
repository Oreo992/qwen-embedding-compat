"""诊断本地 vs 百炼向量差异的根本原因"""

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


def api_embed(client, text, dim=2048, instruct=None):
    payload = {
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"text": text}]},
        "parameters": {"dimension": dim},
    }
    if instruct is not None:
        payload["parameters"]["instruct"] = instruct
    resp = client.post(URL, json=payload)
    resp.raise_for_status()
    return resp.json()["output"]["embeddings"][0]["embedding"]


print("Loading model...")
model = Qwen3VLEmbedder(
    model_name_or_path=r"D:\qwen3-vl-embedding\models\Qwen3-VL-Embedding-8B",
    torch_dtype=torch.bfloat16,
)
print("Model loaded.\n")

text = "hello world"
instruction = "Retrieve video frames relevant to the user's query."

# 本地：4096维原始向量
local_raw = model.process([{"text": text, "instruction": instruction}], normalize=False)
local_raw_vec = local_raw[0].detach().float().cpu().tolist()

# 本地：4096维归一化向量
local_norm = model.process([{"text": text, "instruction": instruction}], normalize=True)
local_norm_vec = local_norm[0].detach().float().cpu().tolist()

print(f"Local raw  dim={len(local_raw_vec)}, norm={np.linalg.norm(local_raw_vec):.4f}")
print(f"Local norm dim={len(local_norm_vec)}, norm={np.linalg.norm(local_norm_vec):.4f}")

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
client = httpx.Client(headers=headers, timeout=30)

# API 不同配置
api_with_inst = api_embed(client, text, 2048, instruction)
api_no_inst = api_embed(client, text, 2048)
api_2560 = api_embed(client, text, 2560, instruction)

print(f"\nAPI (instruct,  dim=2048) norm={np.linalg.norm(api_with_inst):.4f}")
print(f"API (no inst,   dim=2048) norm={np.linalg.norm(api_no_inst):.4f}")
print(f"API (instruct,  dim=2560) norm={np.linalg.norm(api_2560):.4f}")

# 本地截取+归一化
local_trunc_renorm = l2_normalize(local_raw_vec[:2048])
local_norm_trunc = local_norm_vec[:2048]

print("\n=== Cosine 对比 (dim=2048) ===")
print(f"local_raw[:2048]_renorm  vs API(instruct):    {cos(local_trunc_renorm, api_with_inst):.6f}")
print(f"local_raw[:2048]_renorm  vs API(no_inst):     {cos(local_trunc_renorm, api_no_inst):.6f}")
print(f"local_norm[:2048]        vs API(instruct):    {cos(local_norm_trunc, api_with_inst):.6f}")
print(f"local_norm[:2048]        vs API(no_inst):     {cos(local_norm_trunc, api_no_inst):.6f}")
print(f"API(instruct)            vs API(no_inst):     {cos(api_with_inst, api_no_inst):.6f}")

# 对比 2560 维
local_2560_renorm = l2_normalize(local_raw_vec[:2560])
print(f"\n=== Cosine 对比 (dim=2560) ===")
print(f"local_raw[:2560]_renorm  vs API(dim=2560):    {cos(local_2560_renorm, api_2560):.6f}")

# 尝试4096维
try:
    api_4096 = api_embed(client, text, 4096, instruction)
    print(f"\nAPI (dim=4096): dim={len(api_4096)}, norm={np.linalg.norm(api_4096):.4f}")
    print(f"local_raw(4096)_norm     vs API(4096):        {cos(l2_normalize(local_raw_vec), api_4096):.6f}")
except Exception as e:
    print(f"\nAPI dim=4096 failed: {e}")

# 前10个值对比
print("\n=== 前5个值对比 ===")
print(f"Local raw[:2048] renorm: {[round(v,6) for v in local_trunc_renorm[:5]]}")
print(f"API (instruct):          {[round(v,6) for v in api_with_inst[:5]]}")
print(f"API (no instruct):       {[round(v,6) for v in api_no_inst[:5]]}")

# 测试不同 instruction
print("\n=== 不同 instruction 的影响 ===")
instructions_to_test = [
    ("default", instruction),
    ("empty", ""),
    ("chinese", "Represent the user's input."),
    ("none_param", None),
]
local_vecs = {}
api_vecs = {}
for name, inst in instructions_to_test:
    raw = model.process([{"text": text, "instruction": inst or instruction}], normalize=False)
    local_vecs[name] = l2_normalize(raw[0].detach().float().cpu().tolist()[:2048])
    api_vecs[name] = api_embed(client, text, 2048, inst)
    c = cos(local_vecs[name], api_vecs[name])
    print(f"  instruction={name:15s}: local_vs_api={c:.6f}")

# 检查 API 之间的一致性
print("\n=== API 不同 instruction 之间的 cosine ===")
for n1 in api_vecs:
    for n2 in api_vecs:
        if n1 < n2:
            print(f"  API({n1}) vs API({n2}): {cos(api_vecs[n1], api_vecs[n2]):.6f}")

print("\n=== 本地不同 instruction 之间的 cosine ===")
for n1 in local_vecs:
    for n2 in local_vecs:
        if n1 < n2:
            print(f"  Local({n1}) vs Local({n2}): {cos(local_vecs[n1], local_vecs[n2]):.6f}")

client.close()
print("\nDone.")
