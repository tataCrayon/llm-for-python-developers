from huggingface_hub import snapshot_download
from pathlib import Path
import argparse

# 支持的模型及默认本地路径
MODEL_CONFIGS = {
    "bge-large-zh-v1.5": {
        "repo_id": "BAAI/bge-large-zh-v1.5",
        "local_dir": Path("models/bge-large-zh-v1.5")
    },
    "bge-base-zh-v1.5": {
        "repo_id": "BAAI/bge-base-zh-v1.5",
        "local_dir": Path("models/bge-base-zh-v1.5")
    },
    "qwen-4b": {
        "repo_id": "Qwen/Qwen3-Embedding-4B",
        "local_dir": Path("models/Qwen_Qwen3-Embedding-4B")
    },
}
HF_ENDPOINT = "https://hf-mirror.com"

def download_model(repo_id: str, local_dir: Path):
    """下载模型到指定路径。"""
    config_file = local_dir / "config.json"
    if config_file.exists():
        print(f"Model already exists at {local_dir}, skip download.")
        return
    try:
        print(f"Downloading model {repo_id} to {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            endpoint=HF_ENDPOINT,
            max_workers=4
        )
        print(f"Model downloaded successfully to {local_dir}")
    except Exception as e:
        print(f"Failed to download model: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace embedding models.")
    parser.add_argument("--model", type=str, default="bge-large-zh-v1.5", choices=MODEL_CONFIGS.keys(), help="模型简称")
    parser.add_argument("--local_dir", type=str, default=None, help="本地保存路径（可选，默认用推荐目录）")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    repo_id = config["repo_id"]
    local_dir = Path(args.local_dir) if args.local_dir else config["local_dir"]
    download_model(repo_id, local_dir)

if __name__ == "__main__":
    main()

"""

python scripts/download_model.py --model bge-large-zh-v1.5 --local_dir F:/ProgrammingEnvironment/AI/EmbeddingModel/BGE-Large-ZH
python scripts/download_model.py --model bge-base-zh-v1.5 --local_dir F:/ProgrammingEnvironment/AI/EmbeddingModel/BGE-Base-ZH
"""