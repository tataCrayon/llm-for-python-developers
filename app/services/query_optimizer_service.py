from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config.deepseek_llm import deep_seek_chat_model
from app.config.logger import setup_logger

logger = setup_logger(__name__)


class QueryOptimizationResult(BaseModel):
    """
    定义查询优化结果的输出结构
    """
    sub_queries: List[str] = Field(description="优化后的子问题列表")


class QueryOptimizerService:
    """
    查询优化服务，用于将用户复杂或模糊的问题分解为一组清晰、具体、可用于向量数据库检索的子问题。
    """
    MULTI_QUERY_PROMPT_TEMPLATE = """
你是一名世界级的信息检索专家和AI助手，擅长将用户复杂或模糊的问题分解为一组清晰、具体、可用于向量数据库检索的子问题。

# 任务
你的任务是分析给定的【原始用户问题】，并将其分解为多个独立的、更小、更具体的子问题。这些子问题将用于并行地从知识库中检索相关信息。

# 指导原则
1.  **保持核心意图**: 所有生成的子问题必须紧密围绕原始问题的核心目的。不要偏离主题。
2.  **原子化与具体化**: 每个子问题都应该足够具体，最好能被一小段独立的文本（一个知识点）所回答。避免生成宽泛或开放性的问题。
3.  **从不同角度分解**: 尝试从不同的方面来分解原始问题。例如，如果用户问“A和B的区别”，你可以分解为：“A的定义是什么？”、“B的定义是什么？”、“A和B在XX方面的具体差异是什么？”。
4.  **为检索而生**: 想象这些子问题是直接输入到搜索引擎或向量数据库中的。它们应该是陈述事实的疑问句，而不是闲聊。
5.  **覆盖关键信息**: 确保分解后的问题集合能够覆盖回答原始问题所需的所有关键信息点。

# 限制
-   生成的子问题数量应在 2 到 5 个之间。
-   输出必须是严格的JSON格式，一个包含字符串的列表。

# 原始用户问题
{original_question}

# 输出JSON
"""

    def __init__(self, llm: ChatOpenAI | None = None):
        """
        初始化查询优化服务。
        Args:
            llm: 可选,默认使用deepseek-chat模型，实测7b也行
        """
        self.llm = llm or deep_seek_chat_model()

        # 创建一个带有强制JSON输出格式的解析器
        self.parser = JsonOutputParser(pydantic_object=QueryOptimizationResult)

        self.prompt = ChatPromptTemplate.from_template(
            self.MULTI_QUERY_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        # 将Prompt, LLM, 和 Parser链接成一个处理链
        self.chain = self.prompt | self.llm | self.parser

    def optimize_query(self, original_question: str) -> List[str]:
        """
        优化查询并将其拆解为多个子问题。
        Args:
            original_question: 用户的原始查询字符串。
        Returns:
            优化后的子问题列表。
        """
        try:
            result = self.chain.invoke({"original_question": original_question})
            return result['sub_queries']
        except Exception as e:
            # 在生产环境中，这里应该有更完善的日志和错误处理
            logger.error(f"查询优化时发生错误: {e}")
            # 返回空列表作为默认值
            return []
