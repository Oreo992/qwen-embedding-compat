# Qwen 本地 / 百炼向量兼容性采样

这组工具让本地 `Qwen/Qwen3-VL-Embedding-8B` 和百炼
`qwen3-vl-embedding` 处理同一份文字/图片清单，分别输出统一格式的 JSONL。
它们用于验证向量空间兼容性，不会修改现有 ChromaDB。

## 1. 准备输入清单

创建 `compat-inputs.jsonl`，每行必须有唯一 `id`，并且只能有 `text` 或 `image`
其中一个字段。相对图片路径以清单所在目录为基准。

```jsonl
{"id":"frame-001","image":"frames/frame-001.jpg"}
{"id":"frame-002","image":"frames/frame-002.jpg"}
{"id":"query-001","text":"外卖骑手在城市街道上骑车"}
{"id":"query-002","text":"餐厅里朋友们开心聚餐"}
```

先用 20～50 张代表性图片和 10～20 条真实搜索词冒烟；确认脚本无误后，再扩到
500～1000 张图片。

## 2. 本地 GPU 机器

以下命令假定本地已经有官方 Qwen 仓库和 8B 模型环境：

```bash
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
cd Qwen3-VL-Embedding
bash scripts/setup_environment.sh
source .venv/bin/activate
```

把整个 `embedding_compat` 目录和输入清单复制到本地机器，然后执行：

```bash
python embedding_compat/local_qwen_embed.py \
  --qwen-repo /absolute/path/Qwen3-VL-Embedding \
  --model /absolute/path/Qwen3-VL-Embedding-8B \
  --manifest /absolute/path/compat-inputs.jsonl \
  --output /absolute/path/local-8b-2048.jsonl \
  --dimension 2048 \
  --batch-size 1
```

如果模型允许从 Hugging Face 下载，`--model` 也可以直接使用默认的
`Qwen/Qwen3-VL-Embedding-8B`。8B 原始向量是 4096 维；脚本截取前 2048 维后
重新做 L2 归一化，以匹配百炼可请求的维度。

显存充足时可提高 `--batch-size`。发生 OOM 时，已完成的行仍会保留；保持同一
`--output` 再运行即可跳过成功项。使用 `--overwrite` 才会从头开始。

## 3. 能访问百炼的机器

安装 HTTP 客户端并通过环境变量配置密钥：

```bash
python -m pip install httpx
export DASHSCOPE_API_KEY='你的百炼 API Key'
```

执行线上采样：

```bash
python embedding_compat/bailian_qwen_embed.py \
  --manifest /absolute/path/compat-inputs.jsonl \
  --output /absolute/path/bailian-2048.jsonl \
  --model qwen3-vl-embedding \
  --dimension 2048 \
  --concurrency 4
```

API Key 只能从 `DASHSCOPE_API_KEY` 读取，不会写进输出。线上脚本会对 429、5xx
和网络瞬时错误重试；单项失败写成 `status=error`，重跑时只跳过已成功项。

## 4. 必须保持一致的参数

两个命令必须使用相同的：

- `--dimension 2048`
- `--instruction`（默认值已经相同）
- 输入图片原文件
- 输入文字原文

输出中的 `model`、`dimension`、`instruction` 和 `modality` 会随每条向量保存，
便于发现配置漂移。即使输出维度一致，也必须继续比较相同输入的 cosine 和真实
文字查图片的 Top-K；本工具不会预设两个服务一定兼容。

## 5. 离线验证脚本本身

在 `videotest` 仓库根目录运行：

```bash
python -m unittest tools/embedding_compat/test_embedding_compat.py -v
python -m py_compile \
  tools/embedding_compat/io_utils.py \
  tools/embedding_compat/local_qwen_embed.py \
  tools/embedding_compat/bailian_qwen_embed.py
```
