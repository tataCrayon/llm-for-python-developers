import os
from typing import ClassVar, Dict, Any

from langchain_openai import ChatOpenAI

from app.config.logger import setup_logger
from app.listeners import invoke_listener

logger = setup_logger(__name__)

# 小模型配置
KEYWORD_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen-1.5b": {
        "base_url": os.getenv("LOCAL_MODEL_SERVE_BASE_URL", "http://localhost:8000/v1"),
        "model_path": os.getenv("LOCAL_MODEL_SERVE_MODEL_PATH",
                                "./Qwen2.5-1.5B-Instruct/Qwen/Qwen2___5-1___5B-Instruct"),
        "api_key": "EMPTY",
        "temperature": 0.1,
        "max_tokens": 4096,
        "timeout": 600,
    }
}

# 默认模型配置
DEFAULT_KEYWORD_MODEL = "qwen-1.5b"


class KeywordModelComponent:
    """
    支持多小模型的单例加载与管理。
    """
    _instances: ClassVar[Dict[str, "KeywordModelComponent"]] = {}
    _llms: Dict[str, ChatOpenAI]

    def __new__(cls, model_key: str = DEFAULT_KEYWORD_MODEL) -> "KeywordModelComponent":
        if model_key not in cls._instances:
            instance = super().__new__(cls)
            instance._llms = {}
            config = KEYWORD_MODELS.get(model_key)
            if not config:
                raise ValueError(f"Keyword model '{model_key}' not found in KEYWORD_MODELS.")
            instance._llms[model_key] = ChatOpenAI(
                base_url=config["base_url"],
                api_key=config["api_key"],
                model=config["model_path"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                timeout=config["timeout"],
                callbacks=invoke_listener(),
            )
            cls._instances[model_key] = instance
        return cls._instances[model_key]

    def get(self, model_key: str) -> ChatOpenAI:
        if model_key not in self._llms:
            config = KEYWORD_MODELS.get(model_key)
            if not config:
                raise ValueError(f"Keyword model '{model_key}' not found in KEYWORD_MODELS.")
            self._llms[model_key] = ChatOpenAI(
                base_url=config["base_url"],
                api_key=config["api_key"],
                model=config["model_path"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                timeout=config["timeout"],
                callbacks=invoke_listener(),
            )
        return self._llms[model_key]

    @classmethod
    def get_default(cls) -> ChatOpenAI:
        return cls().get(DEFAULT_KEYWORD_MODEL)

    @classmethod
    def available_models(cls) -> list[str]:
        return list(KEYWORD_MODELS.keys())
