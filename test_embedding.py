"""
Qwen3-VL-Embedding-8B 测试脚本
测试文本嵌入和图文相似度计算
"""
import torch
import time
from qwen3_vl_embedding import Qwen3VLEmbedder

MODEL_PATH = r"D:\qwen3-vl-embedding\models\Qwen3-VL-Embedding-8B"

def main():
    print("=" * 60)
    print("Qwen3-VL-Embedding-8B 部署测试")
    print("=" * 60)

    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"\n正在加载模型: {MODEL_PATH}")
    print("使用 bfloat16 精度以节省显存...")
    start = time.time()

    model = Qwen3VLEmbedder(
        model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )

    load_time = time.time() - start
    print(f"模型加载完成，耗时: {load_time:.1f} 秒")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"显存占用: {allocated:.2f} GB (allocated) / {reserved:.2f} GB (reserved)")

    # ===== 测试 1: 纯文本嵌入 =====
    print("\n" + "-" * 40)
    print("测试 1: 纯文本嵌入 & 相似度")
    print("-" * 40)

    text_inputs = [
        {
            "text": "一只小猫在草地上玩耍",
            "instruction": "检索与用户查询相关的文本或图片。",
        },
        {"text": "小猫咪在绿色的草坪上快乐地奔跑和嬉戏"},
        {"text": "今天的股市行情走势如何"},
        {"text": "A kitten playing on the grass"},
    ]

    start = time.time()
    embeddings = model.process(text_inputs)
    infer_time = time.time() - start

    print(f"推理耗时: {infer_time:.2f} 秒")
    print(f"嵌入维度: {embeddings.shape}")

    similarity_matrix = embeddings @ embeddings.T
    print("\n文本相似度矩阵:")
    texts = [
        "一只小猫在草地上玩耍 (query)",
        "小猫咪在绿色草坪上嬉戏",
        "今天股市行情如何",
        "A kitten playing on grass",
    ]
    print(f"{'':>30}", end="")
    for i in range(len(texts)):
        print(f"  [{i}]", end="")
    print()
    for i, t in enumerate(texts):
        print(f"  [{i}] {t:>26}", end="")
        for j in range(len(texts)):
            print(f" {similarity_matrix[i][j].item():.3f}", end="")
        print()

    # ===== 测试 2: 图文跨模态嵌入 =====
    print("\n" + "-" * 40)
    print("测试 2: 图文跨模态嵌入")
    print("-" * 40)

    multimodal_inputs = [
        {
            "text": "一个女人和她的狗在沙滩上玩耍",
            "instruction": "检索与用户查询相关的图片。",
        },
        {
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        },
    ]

    start = time.time()
    mm_embeddings = model.process(multimodal_inputs)
    infer_time = time.time() - start

    print(f"推理耗时: {infer_time:.2f} 秒")
    print(f"嵌入维度: {mm_embeddings.shape}")

    sim = (mm_embeddings[0] @ mm_embeddings[1]).item()
    print(f"\n文本 '一个女人和她的狗在沙滩上玩耍'")
    print(f"  与图片的相似度: {sim:.4f}")

    print("\n" + "=" * 60)
    print("所有测试通过！模型部署成功！")
    print("=" * 60)

    print("\n使用示例:")
    print("""
from qwen3_vl_embedding import Qwen3VLEmbedder
import torch

model = Qwen3VLEmbedder(
    model_name_or_path=r"D:\\qwen3-vl-embedding\\models\\Qwen3-VL-Embedding-8B",
    torch_dtype=torch.bfloat16,
)

# 文本嵌入
inputs = [{"text": "你的文本内容"}]
embeddings = model.process(inputs)

# 图文嵌入
inputs = [
    {"text": "搜索查询", "instruction": "检索相关图片"},
    {"image": "path/to/image.jpg"},
]
embeddings = model.process(inputs)
similarity = embeddings[0] @ embeddings[1]
""")


if __name__ == "__main__":
    main()
