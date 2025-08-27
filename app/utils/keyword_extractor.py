import os
from typing import List

from app.config.logger import setup_logger
from app.config.small_model import KeywordModelComponent

logger = setup_logger(__name__)


class KeywordExtractor:
    def __init__(self, model_key: str = None):
        """
        初始化关键词提取器
        :param model_key: 模型键名，用于从配置中获取模型参数
        """
        # 从配置文件获取模型配置
        if model_key:
            self.llm = KeywordModelComponent().get(model_key)
        else:
            self.llm = KeywordModelComponent.get_default()

    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        从给定文本中提取关键词
        :param text: 输入文本
        :param num_keywords: 需要提取的关键词数量
        :return: 关键词列表
        """
        # 读取提示词模板
        prompt_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "prompts",
                                        "keyword_extraction_prompt.txt")
        prompt_file_path = os.path.abspath(prompt_file_path)
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            logger.error(f"提示词文件 {prompt_file_path} 不存在")
            # 如果文件不存在，使用默认提示词
            prompt_template = "请从以下文本中提取{num_keywords}个最重要的关键词，用逗号分隔：\n\n{text}"

        # 格式化提示词
        prompt = prompt_template.format(num_keywords=num_keywords, text=text)
        logger.debug(f"关键词提取提示词: {prompt}")
        try:
            response = self.llm.invoke(prompt)
            keywords = response.content.strip().split(',')
            return [kw.strip() for kw in keywords if kw.strip()]
        except Exception as e:
            print(f"关键词提取失败: {e}")
            return []
