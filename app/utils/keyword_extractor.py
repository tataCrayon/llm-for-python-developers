import os
from typing import List

from langchain_openai import ChatOpenAI


class KeywordExtractor:
    def __init__(self):
        """
        初始化关键词提取器
        """
        # 从环境变量获取配置
        base_url = os.getenv("LOCAL_MODEL_SERVE_BASE_URL", "http://localhost:8000/v1")
        model_path = os.getenv("LOCAL_MODEL_SERVE_MODEL_PATH", "./Qwen2.5-1.5B-Instruct/Qwen/Qwen2___5-1___5B-Instruct")

        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            model=model_path,
            temperature=0.3,
            max_tokens=100,
            timeout=60,
        )

    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        从给定文本中提取关键词
        :param text: 输入文本
        :param num_keywords: 需要提取的关键词数量
        :return: 关键词列表
        """
        # 读取提示词模板
        prompt_file_path = "data/prompts/keyword_extraction_prompt.txt"
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            # 如果文件不存在，使用默认提示词
            prompt_template = "请从以下文本中提取{num_keywords}个最重要的关键词，用逗号分隔：\n\n{text}"

        # 格式化提示词
        prompt = prompt_template.format(num_keywords=num_keywords, text=text)

        try:
            response = self.llm.invoke(prompt)
            keywords = response.content.strip().split(',')
            return [kw.strip() for kw in keywords if kw.strip()]
        except Exception as e:
            print(f"关键词提取失败: {e}")
            return []
