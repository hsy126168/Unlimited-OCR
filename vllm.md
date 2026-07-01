# Unlimited-OCR vLLM 本地部署指南

## 环境要求

- Python 3.10-3.12
- NVIDIA GPU（建议 24GB+ 显存，如 A100/4090）
- CUDA 12.x 驱动

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

如需从 ModelScope 下载模型：

```bash
pip install modelscope>=1.12.0
```

## 2. 下载模型

### HuggingFace（推荐海外环境）

```bash
python download_model.py --source huggingface --save-dir ./models/Unlimited-OCR
```

国内使用镜像加速：

```bash
python download_model.py --source huggingface --mirror https://hf-mirror.com --save-dir ./models/Unlimited-OCR
```

### ModelScope（推荐国内环境）

```bash
python download_model.py --source modelscope --save-dir ./models/Unlimited-OCR
```

### 自定义模型 ID

```bash
python download_model.py --source huggingface --model-id baidu/Unlimited-OCR
python download_model.py --source modelscope --model-id PaddlePaddle/Unlimited-OCR
```

## 3. 启动 vLLM 推理服务

```bash
python start_vllm.py --model ./models/Unlimited-OCR --gpu 0
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `./models/Unlimited-OCR` | 模型路径或 HuggingFace ID |
| `--gpu` | `0` | CUDA_VISIBLE_DEVICES |
| `--port` | `8000` | 服务端口 |
| `--tensor-parallel-size` | `1` | 多卡并行数 |
| `--max-model-len` | `32768` | 最大上下文长度 |
| `--gpu-memory-utilization` | `0.9` | 显存利用率 |

多卡部署示例：

```bash
python start_vllm.py --model ./models/Unlimited-OCR --gpu 0,1 --tensor-parallel-size 2
```

启动成功后会显示：

```
vLLM server ready (xxs)
  Endpoint: http://127.0.0.1:8000/v1/chat/completions
```

## 4. 启动 FastAPI 解析服务

```bash
python server.py
```

服务运行在 `http://localhost:8080`，提供以下接口：

### POST /parse-pdf

上传 PDF 文件，返回 OCR 解析后的 markdown。

```bash
curl -X POST http://localhost:8080/parse-pdf \
  -F "file=@document.pdf"
```

响应：

```json
{
  "pages": 3,
  "markdown": "# 标题\n\n正文内容..."
}
```

流式返回（SSE）：

```bash
curl -X POST "http://localhost:8080/parse-pdf?stream=true" \
  -F "file=@document.pdf"
```

调整 DPI（影响图片清晰度和推理速度）：

```bash
curl -X POST "http://localhost:8080/parse-pdf?dpi=200" \
  -F "file=@document.pdf"
```

### POST /parse-images

上传一张或多张图片进行 OCR 解析。

```bash
# 单张图片
curl -X POST http://localhost:8080/parse-images \
  -F "files=@page1.png"

# 多张图片（多页模式）
curl -X POST http://localhost:8080/parse-images \
  -F "files=@page1.png" \
  -F "files=@page2.png" \
  -F "files=@page3.png"
```

### GET /health

检查服务状态：

```bash
curl http://localhost:8080/health
```

## 5. Python 客户端调用示例

```python
import requests

# 解析 PDF
with open("document.pdf", "rb") as f:
    resp = requests.post(
        "http://localhost:8080/parse-pdf",
        files={"file": ("document.pdf", f, "application/pdf")},
        params={"dpi": 300},
    )
print(resp.json()["markdown"])

# 解析多张图片
files = [
    ("files", open("page1.png", "rb")),
    ("files", open("page2.png", "rb")),
]
resp = requests.post("http://localhost:8080/parse-images", files=files)
print(resp.json()["markdown"])
```

## 6. 直接调用 vLLM API

不经过 FastAPI，直接请求 vLLM 的 OpenAI 兼容接口：

```python
import base64
import json
import requests

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

payload = {
    "model": "baidu/Unlimited-OCR",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "<image>document parsing."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image('page.png')}"
            }},
        ],
    }],
    "temperature": 0,
    "max_tokens": 32768,
    "skip_special_tokens": False,
    "stream": False,
    # vLLM 自定义参数（通过顶层字段传递）
    "ngram_size": 35,
    "window_size": 128,  # 多页用 1024
}

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
)
print(resp.json()["choices"][0]["message"]["content"])
```

## 关键参数说明

| 参数 | 单页 | 多页/PDF |
|------|------|----------|
| prompt | `<image>document parsing.` | `<image>Multi page parsing.` |
| ngram_size | 35 | 35 |
| window_size | 128 | 1024 |
| skip_special_tokens | False | False |

## 常见问题

### OOM（显存不足）

- 降低 `--gpu-memory-utilization`（如 0.8）
- 降低 `--max-model-len`（如 16384）
- 使用多卡 `--tensor-parallel-size 2`

### 模型加载慢

首次启动需要加载模型权重到 GPU，A100 上约 30-60 秒。后续请求无需重新加载。

### vLLM 版本兼容

Unlimited-OCR 需要 vLLM >= 0.12.0，该版本内置了 `unlimited_ocr` 模型架构和 `NGramPerReqLogitsProcessor`。
