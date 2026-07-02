"""
Start SGLang serving Unlimited-OCR locally.

Usage:
    python start_sglang.py
    python start_sglang.py --model ./models/Unlimited-OCR --gpu 0,1
    python start_sglang.py --port 10000 --gpu 0

Prerequisites:
    pip install sglang[all]
"""

import argparse
import os
import subprocess
import sys
import time

import requests

DEFAULT_MODEL = "./models/Unlimited-OCR"
DEFAULT_PORT = 10000
DEFAULT_HOST = "0.0.0.0"
HEALTH_TIMEOUT = 300
SERVED_MODEL_NAME = "Unlimited-OCR"


def server_ready(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(description="Launch SGLang server for Unlimited-OCR")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or local path")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--page-size", type=int, default=1)
    args = parser.parse_args()

    server_url = f"http://127.0.0.1:{args.port}"
    if server_ready(server_url):
        print(f"SGLang server already running at {server_url}")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model", args.model,
        "--served-model-name", SERVED_MODEL_NAME,
        "--attention-backend", args.attention_backend,
        "--page-size", str(args.page_size),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--context-length", str(args.context_length),
        "--enable-custom-logit-processor",
        "--disable-overlap-schedule",
        "--skip-server-warmup",
        "--host", args.host,
        "--port", str(args.port),
    ]

    print(f"Starting SGLang on GPU {args.gpu}, port {args.port} ...")
    print(f"  Command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, env=env)

    start = time.time()
    while time.time() - start < HEALTH_TIMEOUT:
        if process.poll() is not None:
            print("ERROR: SGLang server exited early.")
            sys.exit(1)
        if server_ready(server_url):
            print(f"SGLang server ready ({time.time() - start:.0f}s)")
            print(f"  Endpoint: {server_url}/v1/chat/completions")
            break
        time.sleep(3)
    else:
        process.terminate()
        print("ERROR: Timed out waiting for SGLang server.")
        sys.exit(1)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down SGLang server...")
        process.terminate()
        process.wait(timeout=30)


if __name__ == "__main__":
    main()
