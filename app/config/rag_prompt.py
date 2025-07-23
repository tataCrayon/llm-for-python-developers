import threading

from pathlib import Path
from app.config.logger import setup_logger
logger = setup_logger(__name__)



class RAGPromptConfig:
    """
    RAG 提示配置类，从文件读取提示内容，实现为单例模式。
    线程安全的单例实现。
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                # 双重检查锁定
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 确保只初始化一次
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                    self.__class__._initialized = True

    def _initialize(self):
        """初始化提示内容"""
        if hasattr(self, 'rag_prompt_path') and self.rag_prompt_path.exists():
            with open(self.rag_prompt_path, 'r', encoding='utf-8') as f:
                self.rag_prompt_content = f.read()
        else:
            self.rag_prompt_content = ""

        logger.info(f"RAG prompts content loaded")
        logger.info(f"RAG prompts content is {self.rag_prompt_content}")

    def refresh(self):
        """刷新提示内容"""
        with self._lock:
            self._initialize()

    @classmethod
    def set_rag_prompt_path(cls, path):
        """设置提示文件路径"""
        instance = cls()
        instance.rag_prompt_path = Path(path)
        instance.refresh()


# 使用示例
if __name__ == "__main__":
    # 设置文件路径并创建实例
    rag_prompt_path = "F:/Projects/PythonProjects/llm-for-python-developers/data/prompts/rag_prompt_zh.txt"
    RAGPromptConfig.set_rag_prompt_path(rag_prompt_path)
    # 获取实例
    config1 = RAGPromptConfig()
    config2 = RAGPromptConfig()

    logger.info(f"是否为同一实例: {config1 is config2}")  # True
    logger.info(f"提示内容: {config1.rag_prompt_content}")

    # 刷新内容
    config1.refresh()

