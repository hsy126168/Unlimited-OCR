"""
Download Unlimited-OCR model from HuggingFace or ModelScope.

Usage:
    python download_model.py --source huggingface
    python download_model.py --source modelscope
    python download_model.py --source huggingface --mirror https://hf-mirror.com
"""

import argparse
import os
import sys


def download_from_huggingface(model_id: str, save_dir: str, mirror: str | None = None):
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"Using HuggingFace mirror: {mirror}")

    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id} from HuggingFace...")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Done: {path}")
    return path


def download_from_modelscope(model_id: str, save_dir: str):
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"Downloading {model_id} from ModelScope...")
    path = snapshot_download(model_id=model_id, cache_dir=save_dir)

    if os.path.exists(path) and os.path.realpath(path) != os.path.realpath(save_dir):
        os.system(f'cp -r "{path}"/* "{save_dir}/"')
        os.system(f'rm -rf "{path}"')

    print(f"Done: {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Download Unlimited-OCR model")
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope"],
        default="huggingface",
    )
    parser.add_argument("--model-id", default=None, help="Override model ID")
    parser.add_argument("--save-dir", default="./models/Unlimited-OCR")
    parser.add_argument(
        "--mirror",
        default=None,
        help="HuggingFace mirror URL (e.g. https://hf-mirror.com)",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.source == "huggingface":
        model_id = args.model_id or "baidu/Unlimited-OCR"
        download_from_huggingface(model_id, args.save_dir, mirror=args.mirror)
    else:
        model_id = args.model_id or "PaddlePaddle/Unlimited-OCR"
        download_from_modelscope(model_id, args.save_dir)


if __name__ == "__main__":
    main()
