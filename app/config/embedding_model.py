from typing import ClassVar, Optional, Dict
import torch

from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from app.config.logger import setup_logger
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger(__name__)

# 推荐模型配置（可扩展）
EMBEDDING_MODELS = {
    "bge-large-zh-v1.5": {
        "model_name": "BAAI/bge-large-zh-v1.5",
        "local_dir": Path(os.getenv("BGE_LARGE_ZH_DIR", "models/bge-large-zh-v1.5")),
        "model_kwargs": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "bge-base-zh-v1.5": {
        "model_name": "BAAI/bge-base-zh-v1.5",
        "local_dir": Path(os.getenv("BGE_BASE_ZH_DIR", "models/bge-base-zh-v1.5")),
        "model_kwargs": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "qwen-4b": {
        "model_name": "Qwen/Qwen3-Embedding-4B",
        "local_dir": Path(os.getenv("QWEN_4B_DIR", "models/Qwen_Qwen3-Embedding-4B")),
        "model_kwargs": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
    },
}

class EmbeddingModelComponent:
    """
    支持多Embedding模型的单例加载与管理。
    """
    _instances: ClassVar[Dict[str, "EmbeddingModelComponent"]] = {}
    _embeddings: Dict[str, HuggingFaceEmbeddings]

    def __new__(cls, model_keys: Optional[list[str]] = None) -> "EmbeddingModelComponent":
        # 参考：我的设备是笔记本，显卡4060 Laptop
        model_keys = model_keys or ["bge-large-zh-v1.5"]
        key_tuple = tuple(sorted(model_keys))
        if key_tuple not in cls._instances:
            instance = super().__new__(cls)
            instance._embeddings = {}
            for key in model_keys:
                config = EMBEDDING_MODELS.get(key)
                if not config:
                    raise ValueError(f"Embedding model '{key}' not found in EMBEDDING_MODELS.")
                local_dir = config["local_dir"]
                if not (local_dir / "config.json").exists():
                    raise RuntimeError(f"模型目录 {local_dir} 不完整，请手动下载模型。")
                logger.info(f"Loading embedding model '{key}' from {local_dir}")
                instance._embeddings[key] = HuggingFaceEmbeddings(
                    model_name=str(local_dir),
                    model_kwargs=config["model_kwargs"],
                    encode_kwargs=config["encode_kwargs"]
                )
            cls._instances[key_tuple] = instance
        return cls._instances[key_tuple]

    def get(self, key: str) -> HuggingFaceEmbeddings:
        if key not in self._embeddings:
            raise ValueError(f"Embedding model '{key}' not loaded.")
        return self._embeddings[key]

    @classmethod
    def get_default(cls) -> HuggingFaceEmbeddings:
        return cls().get("bge-large-zh-v1.5")

    @classmethod
    def available_models(cls) -> list[str]:
        return list(EMBEDDING_MODELS.keys())

