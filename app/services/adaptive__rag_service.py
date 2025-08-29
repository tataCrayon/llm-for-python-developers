import time
from typing import List, Dict

from langchain_core.documents import Document

from app.config.logger import setup_logger
from app.config.rag_prompt import RAGPromptConfig
from app.enum.knowledge_level import KnowledgeLevel
from app.schemas.chat_request import ChatRequest
from app.services.chat_service import DeepSeekService
from app.services.knowledge_level_analysis_service import KnowledgeLevelAnalysisService
from app.services.query_analysis_service import QueryAnalysisService

logger = setup_logger(__name__)


class RAGConfig:
    """RAG配置常量"""
    DEFAULT_K = 20
    MAX_CONTEXT_LENGTH = 20000
    REQUEST_TIMEOUT = 30
    MAX_QUERY_LENGTH = 1000


class ValidationError(Exception):
    """数据验证异常"""
    pass


class RAGTimeoutError(Exception):
    """RAG处理超时异常"""
    pass


class ChatWithRAGService:
    """
    RAG核心服务
    """

    def __init__(self,
                 knowledge_level_analysis_service: KnowledgeLevelAnalysisService | None = None,
                 query_analysis_service: QueryAnalysisService | None = None,
                 llm_service: DeepSeekService | None = None,
                 prompt_config: RAGPromptConfig | None = None,
                 config: RAGConfig | None = None):
        self.knowledge_level_analysis_service = knowledge_level_analysis_service or KnowledgeLevelAnalysisService()
        self.query_analysis_service = query_analysis_service or QueryAnalysisService()
        self.llm_service = llm_service or DeepSeekService()
        self.prompt_config = prompt_config or RAGPromptConfig()
        self.config = config or RAGConfig()

        # 性能监控指标
        self._request_count = 0
        self._total_duration = 0.0

    def _validate_input(self, query: str, documents: List[Document] | None = None) -> None:
        """
        验证输入参数

        Args:
            query: 用户查询
            documents: 相关文档列表
        Returns:
            None
        """
        if not query or not query.strip():
            raise ValidationError("查询不能为空")

        if len(query) > self.config.MAX_QUERY_LENGTH:
            raise ValidationError(f"查询长度不能超过 {self.config.MAX_QUERY_LENGTH} 字符")

        if documents is not None:
            if not isinstance(documents, list):
                raise ValidationError("文档必须是列表类型")

            for i, doc in enumerate(documents):
                if not isinstance(doc, Document):
                    raise ValidationError(f"第 {i + 1} 个文档类型无效")

    def _format_document(self, doc: Document, index: int) -> str:
        """
        格式化单个文档
        Args:
            doc: 文档对象
            index: 文档索引
        Returns:
            格式化后的文档字符串
        """
        return f"""
            文档 {index}:
            来源: {doc.metadata.get('file_name', '未知')} (第{doc.metadata.get('chunk_index', 0)}段)
            关键词: {doc.metadata.get('keywords', '无')}
            内容: {doc.page_content.strip()}
            ---
            """

    def _build_context(self, documents: List[Document] | None = None) -> str:
        """
        构建上下文信息，支持长度限制

        Args:
            documents: 相关文档列表
        Returns:
            上下文字符串
        """
        if not documents:
            return ""

        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            context_part = self._format_document(doc, i)

            # 检查长度限制
            if current_length + len(context_part) > self.config.MAX_CONTEXT_LENGTH:
                logger.warning(f"上下文长度超限，截断到前 {i - 1} 个文档")
                break

            context_parts.append(context_part)
            current_length += len(context_part)

        logger.debug(f"构建上下文完成，包含 {len(context_parts)} 个文档，总长度 {current_length} 字符")
        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str, knowledge_level: KnowledgeLevel) -> str:
        """
        构建最终提示

        Args:
            query: 用户查询
            context: 上下文信息
            knowledge_level: 知识水平
        Returns:
            最终提示词
        """
        return self.prompt_config.rag_prompt_content.format(
            query=query,
            context=context,
            knowledge_level=knowledge_level.value
        )

    def _build_error_response(self, error_message: str, knowledge_level: KnowledgeLevel) -> Dict:
        """
        构建错误响应
        :param error_message: 错误消息
        :param knowledge_level: 知识水平
        :return: 错误响应字典
        """
        return {
            "answer": error_message,
            "knowledge_level": knowledge_level.value,
            "sources": [],
            "context_docs_count": 0,
            "error": True
        }

    def _build_success_response(self, answer: str,
                                knowledge_level: KnowledgeLevel,
                                documents: List[Document] | None = None) -> Dict:
        """
        构建成功响应
        :param answer: 生成的答案
        :param knowledge_level: 知识水平
        :param documents: 源文档列表
        :return: 成功响应字典
        """
        sources = []
        if documents:
            sources = [
                {
                    "file_name": doc.metadata.get('file_name', '未知'),
                    "chunk_index": doc.metadata.get('chunk_index', 0),
                    "keywords": doc.metadata.get('keywords', '')
                }
                for doc in documents
            ]

        return {
            "answer": answer,
            "knowledge_level": knowledge_level.value,
            "sources": sources,
            "context_docs_count": len(documents) if documents else 0,
            "error": False
        }

    async def generate_answer(self, query: str,
                              documents: List[Document] | None = None,
                              knowledge_level: KnowledgeLevel | None = KnowledgeLevel.BEGINNER
                              ) -> Dict:
        """
        生成最终回答

        Args:
            query: 用户查询
            documents: 相关文档列表
            knowledge_level: 知识水平

        Returns:
            回答字典
        """
        start_time = time.time()

        logger.info(f"生成回答，查询: {query}, 知识水平: {knowledge_level.value}")

        try:

            # 输入验证
            self._validate_input(query, documents)

            logger.info(
                f"生成回答开始 - 查询长度: {len(query)}, 文档数: {len(documents or [])}, 知识水平: {knowledge_level.value}")

            # 构建上下文
            context = self._build_context(documents)

            # 构建提示
            prompt = self._build_prompt(query, context, knowledge_level)

            # 调用LLM生成回答
            try:
                answer = await self.llm_service.generate(prompt)
                if not answer or not answer.strip():
                    raise ValueError("LLM返回空响应")

            except Exception as llm_error:
                logger.error(f"LLM调用失败: {llm_error}")
                raise RAGTimeoutError("AI服务暂时不可用，请稍后重试")

            # 构建成功响应
            response = self._build_success_response(answer, knowledge_level, documents)

            duration = time.time() - start_time
            logger.info(f"回答生成完成 - 耗时: {duration:.2f}s, 答案长度: {len(answer)} 字符")

            return response

        except ValidationError as e:
            logger.warning(f"参数验证失败: {e}")
            return self._build_error_response("输入参数无效，请检查您的查询", knowledge_level)
        except RAGTimeoutError as e:
            logger.error(f"RAG处理超时: {e}")
            return self._build_error_response(str(e), knowledge_level)
        except Exception as e:
            logger.error(f"生成回答时出现未知错误: {e}", exc_info=True)
            return self._build_error_response("抱歉，生成回答时出现错误，请稍后重试", knowledge_level)

    async def adaptive_chat(self,
                            request: ChatRequest,
                            k: int | None = None) -> Dict:
        """
        自适应聊天
        Args:
            request: 聊天请求
            k: 检索文档数量
        Returns:
            回答字典
        """
        route, reasoning = self.query_analysis_service.analyze_query(request.user_message)

        if route == 1:
            logger.info(f"判定需要执行RAG流程，原因:{reasoning}")
            return await self.rag_pipeline(request.user_message, k)
        else:
            logger.info(f"判定为LLM直接回答，原因:{reasoning}")
            self.llm_service.generate(request.user_message)

        return await self.rag_pipeline(request.user_message, k)

    async def rag_pipeline(self,
                           query: str,
                           k: int | None = None) -> Dict:
        """
        完整的RAG流程

        Args:
            query: 用户查询
            k: 检索文档数量

        Returns:
            回答字典
        """
        start_time = time.time()
        k = k or self.config.DEFAULT_K

        # 更新请求计数
        self._request_count += 1

        logger.info(
            f"RAG流程开始 - 请求#{self._request_count}, 查询: {query[:100]}{'...' if len(query) > 100 else ''}, k={k}")

        try:

            # 输入验证
            self._validate_input(query)

            if k <= 0 or k > 20:  # 限制k的合理范围
                raise ValidationError("检索数量k必须在1-20之间")

            # 1. 增强检索

            try:
                documents = await self.knowledge_level_analysis_service.enhanced_search(query, k)
            except Exception as search_error:
                logger.error(f"文档检索失败: {search_error}")
                documents = []

            # 2. 确定知识水平
            if not documents:
                logger.info(f"未检索到相关文档，知识水平定义为 BEGINNER ")
                knowledge_level = KnowledgeLevel.BEGINNER
            else:
                try:
                    knowledge_level = await self.knowledge_level_analysis_service.infer_knowledge_level(query,documents)
                except Exception as level_error:
                    logger.warning(f"知识水平推断失败: {level_error}, 使用默认值")
                    knowledge_level = KnowledgeLevel.BEGINNER

            # 3. 生成回答
            response = await self.generate_answer(query, documents, knowledge_level)

            # 4. 记录性能指标
            duration = time.time() - start_time
            self._total_duration += duration
            avg_duration = self._total_duration / self._request_count

            logger.info(f"RAG流程完成 - 耗时: {duration:.2f}s, 平均耗时: {avg_duration:.2f}s")

            # 在响应中添加性能信息（调试模式下）
            if logger.isEnabledFor(10):  # DEBUG level
                response["debug_info"] = {
                    "duration": round(duration, 2),
                    "request_count": self._request_count,
                    "avg_duration": round(avg_duration, 2)
                }

            return response

        except ValidationError as e:
            logger.warning(f"RAG流程验证失败: {e}")
            return self._build_error_response(f"参数验证失败: {str(e)}", KnowledgeLevel.BEGINNER)

        except Exception as e:
            logger.error(f"RAG流程出现未知错误: {e}", exc_info=True)
            return self._build_error_response("处理您的问题时出现系统错误，请稍后重试", KnowledgeLevel.BEGINNER)

    def get_performance_stats(self) -> Dict:
        """
        获取性能统计信息
        :return: 性能统计字典
        """
        avg_duration = self._total_duration / max(self._request_count, 1)

        return {
            "request_count": self._request_count,
            "total_duration": round(self._total_duration, 2),
            "average_duration": round(avg_duration, 2)
        }

    def reset_stats(self):
        """重置性能统计"""
        self._request_count = 0
        self._total_duration = 0.0
        logger.info("性能统计已重置")
