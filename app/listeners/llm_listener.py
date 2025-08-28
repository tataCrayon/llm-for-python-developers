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
        logger.debug("======= LLM 调用开始 =======")
        # 提取LLM名称
        llm_name = serialized.get('kwargs', {}).get('model', '')
        if not llm_name:
            llm_name = serialized.get('kwargs', {}).get('model_name', '')

        # 如果仍然没有获取到，尝试从id数组中解析
        if not llm_name:
            id_array = serialized.get('id', [])
            if id_array:
                for item in reversed(id_array):
                    # 排除明显的类名
                    if item and not item.startswith('langchain') and not item.startswith('Chat'):
                        llm_name = item
                        break
                # 如果没有找到合适的名称，则使用最后一个元素
                if not llm_name:
                    llm_name = id_array[-1]
        if not llm_name:
            llm_name = 'Unknown'

        prompt_str = json.dumps(prompts, ensure_ascii=False, indent=2)
        log_message = f"""
模型名称: {llm_name}
提示词:
{prompt_str}
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