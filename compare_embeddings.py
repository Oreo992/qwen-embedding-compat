"""
阿里云百炼 vs 本地 Qwen3-VL-Embedding-8B 向量一致性对比测试
覆盖场景：纯文本、纯图片、文本-图片跨模态检索、不同维度、不同instruction
"""

import base64
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch

# ─── 配置 ───────────────────────────────────────────────────────
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
MODEL_PATH = r"D:\qwen3-vl-embedding\models\Qwen3-VL-Embedding-8B"
IMAGE_DIR = Path(r"D:\qwen3-vl-embedding\test_images")

DIMENSIONS_TO_TEST = [2048]
DEFAULT_DIM = 2048
DEFAULT_INSTRUCTION = "Retrieve video frames relevant to the user's query."


# ─── 工具函数 ────────────────────────────────────────────────────
def cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def l2_normalize(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vec
    return (arr / norm).tolist()


def truncate_and_normalize(vec: list[float], dim: int) -> list[float]:
    return l2_normalize(vec[:dim])


def image_to_data_uri(path: Path) -> str:
    suffix_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}
    mime = suffix_map.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# ─── 百炼 API 调用 ────────────────────────────────────────────────
def api_embed_text(client: httpx.Client, text: str, dim: int = DEFAULT_DIM,
                   instruction: str | None = None) -> list[float]:
    payload: dict = {
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"text": text}]},
        "parameters": {"dimension": dim},
    }
    if instruction:
        payload["parameters"]["instruct"] = instruction
    resp = client.post(DASHSCOPE_URL, json=payload)
    resp.raise_for_status()
    body = resp.json()
    return body["output"]["embeddings"][0]["embedding"]


def api_embed_image(client: httpx.Client, image_path: Path, dim: int = DEFAULT_DIM,
                    instruction: str | None = None) -> list[float]:
    data_uri = image_to_data_uri(image_path)
    payload: dict = {
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"image": data_uri}]},
        "parameters": {"dimension": dim},
    }
    if instruction:
        payload["parameters"]["instruct"] = instruction
    resp = client.post(DASHSCOPE_URL, json=payload)
    resp.raise_for_status()
    body = resp.json()
    return body["output"]["embeddings"][0]["embedding"]


# ─── 本地模型 ─────────────────────────────────────────────────────
def load_local_model():
    print(f"\n{'='*60}")
    print("加载本地模型...")
    print(f"模型路径: {MODEL_PATH}")
    t0 = time.time()
    sys.path.insert(0, r"D:\qwen3-vl-embedding")
    from qwen3_vl_embedding import Qwen3VLEmbedder
    model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH, torch_dtype=torch.bfloat16)
    print(f"模型加载完成，耗时 {time.time()-t0:.1f}s")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU 显存占用: {mem:.2f} GB")
    return model


def local_embed_text(model, text: str, dim: int = DEFAULT_DIM,
                     instruction: str = DEFAULT_INSTRUCTION) -> list[float]:
    inputs = [{"text": text, "instruction": instruction}]
    raw = model.process(inputs, normalize=False)
    vec = raw[0].detach().float().cpu().tolist()
    return truncate_and_normalize(vec, dim)


def local_embed_image(model, image_path: Path, dim: int = DEFAULT_DIM,
                      instruction: str = DEFAULT_INSTRUCTION) -> list[float]:
    inputs = [{"image": str(image_path), "instruction": instruction}]
    raw = model.process(inputs, normalize=False)
    vec = raw[0].detach().float().cpu().tolist()
    return truncate_and_normalize(vec, dim)


# ─── 测试框架 ─────────────────────────────────────────────────────
class TestResult:
    def __init__(self, name: str, cosine: float, local_norm: float, api_norm: float,
                 dim: int, extra: str = ""):
        self.name = name
        self.cosine = cosine
        self.local_norm = local_norm
        self.api_norm = api_norm
        self.dim = dim
        self.extra = extra


def print_results(results: list[TestResult], title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'测试名称':<40} {'维度':>6} {'Cosine':>10} {'本地L2':>8} {'API L2':>8} {'备注'}")
    print("-" * 80)
    for r in results:
        status = "[OK]" if r.cosine > 0.99 else ("[~]" if r.cosine > 0.95 else "[X]")
        print(f"{r.name:<40} {r.dim:>6} {r.cosine:>10.6f} {r.local_norm:>8.4f} {r.api_norm:>8.4f} {status} {r.extra}")


def run_text_tests(model, client: httpx.Client) -> list[TestResult]:
    """场景1: 纯文本 embedding 对比"""
    test_texts = [
        ("中文短句", "一只小猫在草地上玩耍"),
        ("中文长句", "在这个美丽的春天早晨，阳光透过窗户照进来，温暖的光线洒在桌面上，一杯咖啡的香气弥漫在空气中"),
        ("英文短句", "A cat playing on the grass"),
        ("英文长句", "The beautiful sunset painted the sky with shades of orange and purple as the waves gently lapped against the shore"),
        ("中英混合", "今天的weather很好，适合outdoor activities"),
        ("数字和符号", "2024年GDP增长5.2%，CPI同比上涨0.7%"),
        ("专业术语", "Transformer架构使用self-attention机制实现序列建模"),
        ("语义相近A", "外卖骑手在城市街道上骑车"),
        ("语义相近B", "送餐员骑着电动车穿梭在城市道路中"),
        ("语义不相关", "量子计算机利用量子比特的叠加态进行并行计算"),
        ("极短文本", "猫"),
        ("带特殊字符", "Hello! @#$% 你好~！"),
    ]

    results = []
    for name, text in test_texts:
        try:
            local_vec = local_embed_text(model, text)
            api_vec = api_embed_text(client, text)
            cos = cosine_sim(local_vec, api_vec)
            local_norm = float(np.linalg.norm(local_vec))
            api_norm = float(np.linalg.norm(api_vec))
            results.append(TestResult(f"文本: {name}", cos, local_norm, api_norm, DEFAULT_DIM))
            print(f"  [OK] 文本: {name} -> cosine={cos:.6f}")
        except Exception as e:
            print(f"  [ERR] 文本: {name} -> {e}")
            results.append(TestResult(f"文本: {name}", 0.0, 0.0, 0.0, DEFAULT_DIM, f"ERROR: {e}"))
    return results


def run_image_tests(model, client: httpx.Client) -> list[TestResult]:
    """场景2: 纯图片 embedding 对比"""
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))
    results = []
    for img in image_files:
        try:
            local_vec = local_embed_image(model, img)
            api_vec = api_embed_image(client, img)
            cos = cosine_sim(local_vec, api_vec)
            local_norm = float(np.linalg.norm(local_vec))
            api_norm = float(np.linalg.norm(api_vec))
            results.append(TestResult(f"图片: {img.name}", cos, local_norm, api_norm, DEFAULT_DIM))
            print(f"  [OK] 图片: {img.name} -> cosine={cos:.6f}")
        except Exception as e:
            print(f"  [ERR] 图片: {img.name} -> {e}")
            results.append(TestResult(f"图片: {img.name}", 0.0, 0.0, 0.0, DEFAULT_DIM, f"ERROR: {e}"))
    return results


def run_cross_modal_tests(model, client: httpx.Client) -> list[TestResult]:
    """场景3: 跨模态文本-图片检索一致性
    验证本地和API在做文图匹配时，相似度排序是否一致
    """
    queries = [
        "一个红色的圆形",
        "一个蓝色的方形",
        "自然风景画面，有天空和草地",
        "一只橙色的猫",
        "图片中有文字",
    ]
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))

    results = []
    for qi, query in enumerate(queries):
        local_query_vec = local_embed_text(model, query)
        api_query_vec = api_embed_text(client, query)

        local_scores = []
        api_scores = []
        for img in image_files:
            local_img_vec = local_embed_image(model, img)
            api_img_vec = api_embed_image(client, img)

            local_scores.append((img.name, cosine_sim(local_query_vec, local_img_vec)))
            api_scores.append((img.name, cosine_sim(api_query_vec, api_img_vec)))

        local_rank = [x[0] for x in sorted(local_scores, key=lambda x: -x[1])]
        api_rank = [x[0] for x in sorted(api_scores, key=lambda x: -x[1])]
        rank_match = local_rank == api_rank
        top1_match = local_rank[0] == api_rank[0]

        cos_query = cosine_sim(local_query_vec, api_query_vec)
        extra = f"排序{'完全一致' if rank_match else ('Top1一致' if top1_match else '排序不同')}"
        extra += f" | 本地Top1={local_rank[0]} API_Top1={api_rank[0]}"
        results.append(TestResult(f"跨模态: {query[:20]}", cos_query, 0, 0, DEFAULT_DIM, extra))

        print(f"  [{'OK' if rank_match else 'WARN'}] 跨模态 query='{query[:25]}' 排序{'一致' if rank_match else '不同'}")
        print(f"    本地排序: {local_rank}")
        print(f"    API排序:  {api_rank}")
        for name, ls in local_scores:
            as_score = next(s for n, s in api_scores if n == name)
            print(f"      {name}: 本地={ls:.4f} API={as_score:.4f} diff={abs(ls-as_score):.4f}")

    return results


def run_instruction_tests(model, client: httpx.Client) -> list[TestResult]:
    """场景4: 不同 instruction 对 embedding 的影响"""
    instructions = [
        ("默认instruction", DEFAULT_INSTRUCTION),
        ("中文instruction", "检索与用户查询相关的图片。"),
        ("通用instruction", "Represent the user's input."),
        ("无instruction", ""),
    ]
    text = "外卖骑手在城市街道上骑车"
    results = []

    base_local = None
    base_api = None
    for name, inst in instructions:
        try:
            local_vec = local_embed_text(model, text, instruction=inst or DEFAULT_INSTRUCTION)
            api_vec = api_embed_text(client, text, instruction=inst if inst else None)
            cos = cosine_sim(local_vec, api_vec)
            local_norm = float(np.linalg.norm(local_vec))
            api_norm = float(np.linalg.norm(api_vec))

            extra = ""
            if base_local is not None:
                local_drift = cosine_sim(local_vec, base_local)
                api_drift = cosine_sim(api_vec, base_api)
                extra = f"本地偏移={1-local_drift:.6f} API偏移={1-api_drift:.6f}"

            if base_local is None:
                base_local = local_vec
                base_api = api_vec

            results.append(TestResult(f"Instruction: {name}", cos, local_norm, api_norm, DEFAULT_DIM, extra))
            print(f"  [OK] {name} -> cosine={cos:.6f} {extra}")
        except Exception as e:
            print(f"  [ERR] {name} -> {e}")
            results.append(TestResult(f"Instruction: {name}", 0.0, 0.0, 0.0, DEFAULT_DIM, f"ERROR: {e}"))
    return results


def run_dimension_tests(model, client: httpx.Client) -> list[TestResult]:
    """场景5: 不同维度下的一致性"""
    dims = [256, 512, 768, 1024, 2048]
    text = "一只小猫在草地上玩耍"
    results = []

    for dim in dims:
        try:
            local_vec = local_embed_text(model, text, dim=dim)
            api_vec = api_embed_text(client, text, dim=dim)
            cos = cosine_sim(local_vec, api_vec)
            local_norm = float(np.linalg.norm(local_vec))
            api_norm = float(np.linalg.norm(api_vec))
            results.append(TestResult(f"维度测试 dim={dim}", cos, local_norm, api_norm, dim))
            print(f"  [OK] dim={dim} -> cosine={cos:.6f}")
        except Exception as e:
            print(f"  [ERR] dim={dim} -> {e}")
            results.append(TestResult(f"维度测试 dim={dim}", 0.0, 0.0, 0.0, dim, f"ERROR: {e}"))
    return results


def run_semantic_consistency_tests(model, client: httpx.Client) -> list[TestResult]:
    """场景6: 语义一致性 —— 相似/不相似文本对在两端的排序是否一致"""
    pairs = [
        ("语义近", "外卖骑手在城市街道上骑车", "送餐员骑着电动车穿梭在城市道路中"),
        ("语义远", "外卖骑手在城市街道上骑车", "量子计算机利用量子比特进行并行计算"),
        ("跨语言近", "一只小猫在草地上玩耍", "A kitten playing on the grass"),
        ("跨语言远", "一只小猫在草地上玩耍", "Quantum computing uses qubits"),
        ("同义改写", "餐厅里朋友们开心聚餐", "一群好友在饭店里快乐地吃饭"),
    ]
    results = []
    for tag, a, b in pairs:
        try:
            la = local_embed_text(model, a)
            lb = local_embed_text(model, b)
            aa = api_embed_text(client, a)
            ab = api_embed_text(client, b)
            local_sim = cosine_sim(la, lb)
            api_sim = cosine_sim(aa, ab)
            diff = abs(local_sim - api_sim)
            cos_a = cosine_sim(la, aa)
            cos_b = cosine_sim(lb, ab)
            extra = f"本地sim={local_sim:.4f} API_sim={api_sim:.4f} diff={diff:.4f}"
            results.append(TestResult(f"语义: {tag}", cos_a, 0, 0, DEFAULT_DIM, extra))
            print(f"  [OK] {tag}: 本地sim={local_sim:.4f} API_sim={api_sim:.4f} cos_a={cos_a:.6f} cos_b={cos_b:.6f}")
        except Exception as e:
            print(f"  [ERR] {tag}: {e}")
            results.append(TestResult(f"语义: {tag}", 0.0, 0.0, 0.0, DEFAULT_DIM, f"ERROR: {e}"))
    return results


def run_reproducibility_test(model, client: httpx.Client) -> list[TestResult]:
    """场景7: 重复性 —— 多次调用同一输入，向量是否完全一致"""
    text = "外卖骑手在城市街道上骑车"
    results = []

    local_vecs = [local_embed_text(model, text) for _ in range(3)]
    api_vecs = [api_embed_text(client, text) for _ in range(3)]

    for i in range(1, 3):
        lc = cosine_sim(local_vecs[0], local_vecs[i])
        ac = cosine_sim(api_vecs[0], api_vecs[i])
        extra = f"本地重复cosine={lc:.8f} API重复cosine={ac:.8f}"
        results.append(TestResult(f"重复性: run0 vs run{i}", cosine_sim(local_vecs[i], api_vecs[i]),
                                  0, 0, DEFAULT_DIM, extra))
        print(f"  [OK] run0 vs run{i}: 本地={lc:.8f} API={ac:.8f}")

    return results


# ─── 主函数 ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  阿里云百炼 vs 本地 Qwen3-VL-Embedding-8B 一致性对比")
    print("=" * 60)

    model = load_local_model()
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    all_results = []
    with httpx.Client(headers=headers, timeout=60.0) as client:

        print(f"\n{'='*60}")
        print("场景 1/7: 纯文本 embedding 对比")
        print("=" * 60)
        all_results.extend(run_text_tests(model, client))

        print(f"\n{'='*60}")
        print("场景 2/7: 纯图片 embedding 对比")
        print("=" * 60)
        all_results.extend(run_image_tests(model, client))

        print(f"\n{'='*60}")
        print("场景 3/7: 跨模态文图检索排序对比")
        print("=" * 60)
        all_results.extend(run_cross_modal_tests(model, client))

        print(f"\n{'='*60}")
        print("场景 4/7: 不同 Instruction 对比")
        print("=" * 60)
        all_results.extend(run_instruction_tests(model, client))

        print(f"\n{'='*60}")
        print("场景 5/7: 不同维度对比")
        print("=" * 60)
        all_results.extend(run_dimension_tests(model, client))

        print(f"\n{'='*60}")
        print("场景 6/7: 语义一致性对比")
        print("=" * 60)
        all_results.extend(run_semantic_consistency_tests(model, client))

        print(f"\n{'='*60}")
        print("场景 7/7: 重复性对比")
        print("=" * 60)
        all_results.extend(run_reproducibility_test(model, client))

    # ─── 汇总报告 ─────────────────────────────────────────────
    print_results(all_results, "全部测试结果汇总")

    high = [r for r in all_results if r.cosine > 0.99]
    medium = [r for r in all_results if 0.95 < r.cosine <= 0.99]
    low = [r for r in all_results if r.cosine <= 0.95 and r.cosine > 0]
    errors = [r for r in all_results if r.cosine == 0.0 and "ERROR" in r.extra]

    print(f"\n{'='*60}")
    print("  总结")
    print(f"{'='*60}")
    print(f"  总测试数:   {len(all_results)}")
    print(f"  高度一致 (cosine > 0.99):  {len(high)}")
    print(f"  基本一致 (0.95 < cos <= 0.99): {len(medium)}")
    print(f"  明显不同 (cosine <= 0.95): {len(low)}")
    print(f"  执行失败:   {len(errors)}")

    if all_results:
        cosines = [r.cosine for r in all_results if r.cosine > 0]
        if cosines:
            print(f"\n  Cosine 均值: {np.mean(cosines):.6f}")
            print(f"  Cosine 最小: {np.min(cosines):.6f}")
            print(f"  Cosine 最大: {np.max(cosines):.6f}")
            print(f"  Cosine 标准差: {np.std(cosines):.6f}")

    verdict = "基本一致" if len(high) + len(medium) > len(low) else "存在差异"
    print(f"\n  结论: 本地模型与百炼 API 的向量空间 【{verdict}】")
    print(f"{'='*60}")

    output_path = Path(r"D:\qwen3-vl-embedding\comparison_results.json")
    output_data = []
    for r in all_results:
        output_data.append({
            "name": r.name, "cosine": r.cosine,
            "local_norm": r.local_norm, "api_norm": r.api_norm,
            "dim": r.dim, "extra": r.extra,
        })
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
