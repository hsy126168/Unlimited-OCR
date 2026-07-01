"""
Start vLLM serving Unlimited-OCR locally (no Docker).

Usage:
    python start_vllm.py
    python start_vllm.py --model /path/to/local/Unlimited-OCR --gpu 0,1
    python start_vllm.py --port 8000 --gpu 0

Prerequisites:
    pip install -r requirements.txt
"""

import argparse
import os
import subprocess
import sys
import time

import requests

DEFAULT_MODEL = "./models/Unlimited-OCR"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
HEALTH_TIMEOUT = 300


def server_ready(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server for Unlimited-OCR")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or local path")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    server_url = f"http://127.0.0.1:{args.port}"
    if server_ready(server_url):
        print(f"vLLM server already running at {server_url}")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--served-model-name", "baidu/Unlimited-OCR",
        "--trust-remote-code",
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--logits-processors",
        "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
    ]

    print(f"Starting vLLM on GPU {args.gpu}, port {args.port} ...")
    print(f"  Command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, env=env)

    start = time.time()
    while time.time() - start < HEALTH_TIMEOUT:
        if process.poll() is not None:
            print("ERROR: vLLM server exited early.")
            sys.exit(1)
        if server_ready(server_url):
            print(f"vLLM server ready ({time.time() - start:.0f}s)")
            print(f"  Endpoint: {server_url}/v1/chat/completions")
            break
        time.sleep(3)
    else:
        process.terminate()
        print("ERROR: Timed out waiting for vLLM server.")
        sys.exit(1)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
        process.terminate()
        process.wait(timeout=30)


if __name__ == "__main__":
    main()
