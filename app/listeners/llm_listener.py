import json
from typing import List, Dict, Any

from langchain.callbacks.base import BaseCallbackHandler

from app.config.logger import setup_logger

logger = setup_logger(__name__)


class LLMInvokeListener(BaseCallbackHandler):
    """
    自定义回调处理器，用于监听 LLM 的开始和结束事件。
    继承自 langchain 的 BaseCallbackHandler 类。
    """

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        # 提取LLM名称
        llm_name = serialized.get('kwargs', {}).get('model', 'Unknown')
        if llm_name == 'Unknown':
            # 尝试从其他可能的位置获取模型名称
            llm_name = serialized.get('id', ['Unknown'])[-1]

        # 对提示词进行预处理，使其在日志中更易读
        formatted_prompts = [prompt.replace('\n', '\n') for prompt in prompts]

        # 创建更美观的日志格式
        prompt_str = json.dumps(formatted_prompts, ensure_ascii=False, indent=2)
        log_message = f"""
        ======= LLM 调用开始 =======
        模型名称: {llm_name}
        提示词:
        {prompt_str}
        ========================
        """
        logger.debug(log_message)

    def on_llm_end(self, response, **kwargs):
        logger.debug("======= LLM 调用结束 =======")


def invoke_listener() -> List[BaseCallbackHandler]:
    """
    获取所有 listeners 的列表。

    Returns:
        List[BaseCallbackHandler]: listeners 列表
    """
    listeners = []
    tools_call_listener = LLMInvokeListener()
    listeners.append(tools_call_listener)
    return listeners
