from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config.deepseek_llm import deep_seek_chat_model
from app.config.logger import setup_logger

logger = setup_logger(__name__)


class QueryAnalysisResult(BaseModel):
    """
    定义查询分析结果的输出结构
    """
    route: Literal[0, 1] = Field(description="查询的路由决策。0代表直接由通用LLM回答，1代表需要通过RAG流程处理。")
    reasoning: str = Field(description="做出此路由决策的简要理由。")


class QueryAnalysisService:
    """
    查询分析服务，用于判断用户查询是否需要经过RAG流程。
    """
    ROUTER_PROMPT_TEMPLATE = """
你是一个高级的AI查询路由专家。你的任务是分析用户的查询，并决定该查询应该由一个【通用大模型】直接回答，还是应该路由到一个【基于用户个人知识库的RAG系统】进行处理。

# 路由决策标准:
你必须根据以下标准，决定返回 0 还是 1：

## 返回 1 (路由到RAG系统):
当查询满足以下任何一个条件时，应返回 1：
1.  **依赖个人知识**: 查询明确或隐含地需要访问用户的个人笔记、文档、项目数据等私有知识库。
2.  **深度专业问题**: 查询涉及非常具体、深入的专业领域知识，这些知识不太可能是公开的常识，而更可能存在于用户的个人知识库中。
3.  **个性化需求**: 查询需要结合用户的知识水平、学习历史或个人背景来进行回答。
4.  **上下文关联**: 查询是关于用户之前讨论过的话题，需要历史上下文才能准确回答。

## 返回 0 (由通用大模型直接回答):
当查询属于以下类别时，应返回 0：
1.  **通用知识**: 查询的是公开的、普遍性的知识或常识 (例如："法国的首都是哪里？", "解释一下牛顿第一定律")。
2.  **创造性任务**: 要求进行内容创作 (例如："写一首诗", "编一个故事")。
3.  **通用编程/技术问题**: (例如："Python的list和tuple有什么区别？")。
4.  **闲聊与指令**: 简单的问候、闲聊或不涉及知识查询的指令 (例如："你好", "你叫什么名字？")。
5.  **数学计算**: 执行数学运算。

# 示例:
- 查询: "根据我上次的笔记，总结一下RAG和Finetune的区别" -> 返回: 1 (理由: 明确依赖“我的笔记”)
- 查询: "RAG和Finetune有什么区别？" -> 返回: 1 (理由: 这是一个专业领域对比问题，很可能需要用户知识库中的精确定义和上下文来获得最佳答案)
- 查询: "LangChain的Agent ReAct框架是怎么工作的？" -> 返回: 1 (理由: 深度专业问题，用户的知识库可能包含更具体、个性化的理解和实例)
- 查询: "你好吗？" -> 返回: 0 (理由: 属于闲聊)
- 查询: "帮我写一个Python的快速排序算法" -> 返回: 0 (理由: 通用编程问题，无需个人知识库)

# 你的任务:
分析下面的【用户查询】，并严格按照指定的JSON格式输出你的决策。

用户查询: "{query}"

{format_instructions}
"""

    def __init__(self, llm=None):
        """
        初始化查询分析服务。
        Args:
            llm: 可选，一个兼容LangChain的语言模型实例。如果未提供，将使用默认的GPT-4-turbo。
        """
        # 为了保证路由的准确性，建议使用能力较强的模型
        self.llm = llm or deep_seek_chat_model()

        # 创建一个带有强制JSON输出格式的解析器
        self.parser = JsonOutputParser(pydantic_object=QueryAnalysisResult)

        self.prompt = ChatPromptTemplate.from_template(
            self.ROUTER_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        # 将Prompt, LLM, 和 Parser链接成一个处理链
        self.chain = self.prompt | self.llm | self.parser

    def analyze_query(self, query: str) -> tuple[int, str]:
        """
        分析查询并决定是否需要走RAG流程。
        Args:
            query: 用户的查询字符串。
        Returns:
            一个元组 (route, reasoning)，其中:
            - route (int): 0 表示不需要走RAG，1 表示需要走RAG。
            - reasoning (str): 做出该决策的理由。
        """
        try:
            result = self.chain.invoke({"query": query})
            logger.info(f"用户Query分析结果: {result}")
            return result['route'], result['reasoning']
        except Exception as e:
            # 在生产环境中，这里应该有更完善的日志和错误处理
            logger.info(f"查询分析时发生错误: {e}")
            # 默认回退到最安全的路径，即走RAG流程
            return 1, "分析时发生错误，默认执行RAG流程。"
