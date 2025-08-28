import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.config.logger import setup_logger
from app.listeners.llm_listener import invoke_listener

load_dotenv()

logger = setup_logger(__name__)

# LLM参数
TEMPERATURE = 0.7
DEEP_SEEK_CHAT_MODEL_NAME = "deepseek-chat"
DEEP_SEEK_REASONER_MODEL_NAME = "deepseek-reasoner"
DEEP_SEEK_API_KEY_NAME = "DEEPSEEK_API_KEY"
DEEP_SEEK_BASE_URL = "https://api.deepseek.com/v1"

def get_deepseek_api_key() -> str:
    """
    从环境变量中获取 DeepSeek API 密钥。

    Raises:
        RuntimeError: 当 DEEPSEEK_API_KEY 环境变量未设置或为空时抛出此异常。

    Returns:
        str: DeepSeek API 密钥。
    """
    api_key = os.getenv(DEEP_SEEK_API_KEY_NAME)
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 环境变量未设置或为空")
    return api_key

def deep_seek_chat_model() -> ChatOpenAI:
    """
    初始化并返回 DeepSeek 聊天模型实例。

    Returns:
        ChatOpenAI: deepseek-chat 模型实例
    """
    api_key = get_deepseek_api_key()
    listeners = invoke_listener()
    model = ChatOpenAI(
        openai_api_key=api_key,
        model_name=DEEP_SEEK_CHAT_MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_base=DEEP_SEEK_BASE_URL,
        callbacks=listeners,
        verbose=True
    )
    return model

def deep_seek_r1_model() -> ChatOpenAI:
    """
    初始化并返回 DeepSeek 聊天模型实例。

    Returns:
        ChatOpenAI: deepseek-reasoner 模型实例
    """
    api_key = get_deepseek_api_key()
    listeners = invoke_listener()
    model = ChatOpenAI(
        openai_api_key=api_key,
        model_name=DEEP_SEEK_REASONER_MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_base=DEEP_SEEK_BASE_URL,
        callbacks=listeners,
        verbose=True
    )
    return model