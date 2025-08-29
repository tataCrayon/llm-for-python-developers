import json
import logging
import re
from typing import List, Dict, Tuple

from langchain_core.documents import Document

from app.config.logger import setup_logger
from app.config.rag_prompt import RAGPromptConfig
from app.db.vector_store.chroma_store import ChromaVectorStore
from app.enum.knowledge_level import KnowledgeLevel
from app.services.chat_service import DeepSeekService
from app.services.query_optimizer_service import QueryOptimizerService

logger = setup_logger(__name__)


class KnowledgeLevelAnalysisService:
    """
    知识水平分析服务
    """

    def __init__(self,
                 vector_store: ChromaVectorStore = None,
                 llm_service: DeepSeekService = None,
                 optimizer_service: QueryOptimizerService | None = None,
                 prompt_config: RAGPromptConfig = None):
        """
        初始化知识水平分析服务
        Args:
            vector_store: 向量存储实例
            llm_service: LLM服务实例
            prompt_config: 提示词配置
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm_service = llm_service or DeepSeekService()
        self.optimizer_service = optimizer_service or QueryOptimizerService()
        self.prompt_config = prompt_config or RAGPromptConfig()

        # 技术术语词典
        self._technical_terms_cache = None

    def _get_technical_terms(self) -> Dict[str, List[str]]:
        """获取技术术语词典（带缓存）"""
        if self._technical_terms_cache is None:
            self._technical_terms_cache = {
                "basic": ["什么是", "介绍", "基础", "入门", "简单解释", "概念", "定义"],
                "intermediate": ["如何", "怎么", "步骤", "方法", "实现", "使用", "配置", "设置"],
                "advanced": ["优化", "性能", "原理", "算法", "架构", "源码", "底层", "设计模式",
                           "embeddings", "cosine similarity", "vector database", "RAG",
                           "transformer", "attention", "fine-tuning", "并发", "异步", "分布式"]
            }
        return self._technical_terms_cache


    def _filter_low_quality_docs(self, docs: List[Document]) -> List[Document]:
        """
        过滤低质量文档
        Args:
            docs: 原始文档列表

        Returns:
            过滤后的文档列表
        """
        if not docs:
            return []

        quality_docs = []

        for doc in docs:
            content = doc.page_content.strip()

            # 改进的过滤条件
            if self._is_low_quality_doc(content):
                logger.debug(f"过滤低质量文档: {content[:50]}...")
                continue

            quality_docs.append(doc)

        return quality_docs

    def _is_low_quality_doc(self, content: str) -> bool:
        """
        判断文档是否为低质量

        Args:
            content: 文档内容

        Returns:
            是否为低质量文档
        """
        # 基本长度检查
        if len(content) < 30:
            return True

        # 纯标题检查
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        if len(non_empty_lines) == 1 and re.match(r'^#{1,6}\s+', non_empty_lines[0]):
            return True

        # 内容密度检查
        word_count = len(content.split())
        if word_count < 5:
            return True

        # 有意义内容检查（避免只有标点符号或特殊字符）
        meaningful_chars = sum(1 for c in content if c.isalnum() or c in '，。？！；：')
        if meaningful_chars / len(content) < 0.5:
            return True

        return False

    async def enhanced_search(self, query: str, k: int| None = 5) -> List[Document]:
        """
        增强搜索函数

        Args:
            query: 搜索查询
            k: 返回的文档数量

        Returns:
            相关文档列表
        """
        if not query or not query.strip():
            logger.warning("搜索查询为空")
            return []

        sub_query_list = self.optimizer_service.optimize_query(query)

        # 如果没有生成子查询，则使用原始查询
        if not sub_query_list:
            sub_query_list = [query]

        logger.info(f"执行增强搜索，原始查询: '{query}', 优化后查询: {sub_query_list}, 请求数量: {k}")

        try:
            # 1. 基础向量搜索，获取更多候选，且去重
            search_k = max(k * 3, 10)  # 确保有足够的候选
            all_results = []

            for sub_query in sub_query_list:
                results = self.vector_store.search_documents(sub_query, k=search_k)
                all_results.extend(results)
                logger.info(f"子查询 '{sub_query}' 返回 {len(results)} 个文档")

            if not all_results:
                logger.warning("向量搜索未返回任何结果")
                return []

            unique_docs = {}
            for doc in all_results:
                # 使用文档内容作为键进行去重
                content = doc.page_content
                if content not in unique_docs:
                    unique_docs[content] = doc
                else:
                    # 如果文档已存在，保留元数据中权重更高的
                    existing_doc = unique_docs[content]
                    if doc.metadata.get('weight', 0.8) > existing_doc.metadata.get('weight', 0.8):
                        unique_docs[content] = doc

            unique_results = list(unique_docs.values())
            logger.info(f"去重后剩余 {len(unique_results)} 个唯一文档")


            # 2. 质量过滤
            quality_results = self._filter_low_quality_docs(unique_results)

            if not quality_results:
                logger.warning("质量过滤后无有效文档")
                return []

            # 3. 重新排序逻辑
            scored_results = self._score_and_rank_documents(quality_results, query)

            # TODO 计划引入BM25

            # 4. 按分数排序并返回top-k
            final_results = [doc for doc, _ in scored_results[:k]]

            logger.info(f"搜索完成，返回 {len(final_results)} 个文档")
            return final_results

        except Exception as e:
            logger.error(f"搜索过程中出错: {e}", exc_info=True)
            return []

    def _score_and_rank_documents(self, docs: List[Document], query: str) -> List[Tuple[Document, float]]:
        """
        对文档进行评分和排序

        Args:
            docs: 文档列表
            query: 查询字符串

        Returns:
            (文档, 分数)的元组列表，按分数降序排列
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored_docs = []

        for doc in docs:
            score = self._calculate_relevance_score(doc, query_lower, query_terms)
            scored_docs.append((doc, score))

        # 按分数降序排列
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 记录排序结果
        if logger.isEnabledFor(logging.DEBUG):
            for i, (doc, score) in enumerate(scored_docs[:3]):
                logger.debug(f"排名 {i + 1}: 分数 {score:.3f}, 内容预览: {doc.page_content[:100]}")

        return scored_docs


    def _calculate_relevance_score(self, doc: Document, query_lower: str, query_terms: set) -> float:
        """
        计算文档相关性分数

        Args:
            doc: 文档对象
            query_lower: 小写查询字符串
            query_terms: 查询词集合

        Returns:
            相关性分数
        """
        # 基础权重分数
        base_score = float(doc.metadata.get('weight', 0.8))

        content_lower = doc.page_content.lower()
        content_terms = set(content_lower.split())

        # 1. 关键词匹配分数
        keyword_score = 0.0
        keywords = doc.metadata.get('keywords', '').split(',')
        matched_keywords = sum(1 for kw in keywords if kw.lower().strip() in query_lower)
        if keywords and matched_keywords > 0:
            keyword_score = 0.3 * (matched_keywords / len(keywords))

        # 2. 内容匹配分数
        content_score = 0.0
        matched_terms = query_terms & content_terms
        if query_terms:
            content_score = 0.4 * (len(matched_terms) / len(query_terms))

        # 3. 文档类型权重
        doc_type_score = 0.0
        if re.match(r'^#{1,3}\s+', doc.page_content):  # 重要章节
            doc_type_score = 0.1
        elif '```' in doc.page_content:  # 包含代码示例
            doc_type_score = 0.15

        # 4. 文档长度调整（避免过短或过长文档得分过高）
        length_score = 0.0
        content_length = len(doc.page_content)
        if 100 <= content_length <= 1000:
            length_score = 0.1
        elif content_length > 1000:
            length_score = 0.05

        total_score = base_score + keyword_score + content_score + doc_type_score + length_score
        return min(total_score, 2.0)  # 限制最大分数

    def _get_knowledge_level_analysis_prompt(self, query: str, documents: List[Document]) -> str:
        """
        构建知识水平分析的提示词

        Args:
            query: 用户查询
            documents: 相关文档列表

        Returns:
            分析提示词
        """
        logger.info(f"构建知识水平分析提示词，文档数量: {len(documents)}")

        # 提取文档信息，限制长度避免提示词过长
        doc_info = []
        user_background_info = []
        for i, doc in enumerate(documents[:20], 1):  # 限制最多20个文档
            # 提取更多有用的元数据
            content_preview = doc.page_content[:300]  # 增加预览长度
            if len(doc.page_content) > 300:
                content_preview += "..."

            read_documents = doc.metadata.get('file_name', '')
            user_background_info.append(read_documents)

            doc_summary = {
                "doc_id": i,
                "weight": doc.metadata.get('weight', 0.8),
                "keywords": doc.metadata.get('keywords', ''),
                "content_length": len(doc.page_content),
                "has_code": '```' in doc.page_content,
                "has_heading": bool(re.match(r'^#{1,6}\s+', doc.page_content)),
                "content_preview": content_preview
            }
            doc_info.append(doc_summary)

        user_background = json.dumps(user_background_info, ensure_ascii=False, indent=2)
        doc_context = json.dumps(doc_info, ensure_ascii=False, indent=2)
        return f"""
        你是一个专业的知识水平分析助手。请根据用户查询和检索到的文档内容，分析用户对该主题的知识水平。
        
        **用户背景信息：**
            {user_background}
                       
        **用户查询：**
            {query}

        **检索到的文档信息：**
            {doc_context}
        
        **分析维度：**
            1. **查询复杂度分析**：
               - 是否包含专业术语和技术概念
               - 问题的抽象程度和技术深度
               - 查询的具体性（基础概念 vs 实现细节 vs 优化方案）
               
            2. **文档内容分析**：
               - 检索到的文档技术深度和复杂程度
               - 文档权重分布（反映内容重要性）
               - 是否包含代码示例、实现细节
               
            3. **知识匹配度**：
               - 用户查询与高质量文档的匹配程度
               - 所需知识的层次（概念理解 vs 实践应用 vs 深度原理）
               
            4. **用户背景分析**：
               - 用户已积累的笔记文档，反映其知识积累

        **请按以下JSON格式返回分析结果：**
        ```json
            {{
                "knowledge_level": "beginner|intermediate|advanced",
                "confidence": 0.85,
                "reasoning": {{
                    "query_indicators": ["从查询中识别的知识水平指标"],
                    "document_indicators": ["从文档内容识别的复杂程度指标"],
                    "technical_terms": ["识别到的关键技术术语"],
                    "complexity_score": 0.75
                }},
                "explanation": "详细的推理过程和判断依据"
            }}
        ```

        **知识水平定义：**
            - **beginner**: 对主题陌生，需要基础概念和入门指导
            - **intermediate**: 有基础了解，寻求具体实现方法和实践指导  
            - **advanced**: 深度掌握基础，关注优化、原理和高级应用

        请仔细分析并返回准确的JSON格式结果。
        """

    async def infer_knowledge_level_with_llm(self, query: str, documents: List[Document]) -> KnowledgeLevel:
        """
        使用LLM智能推断用户知识水平

        Args:
            query: 用户查询
            documents: 相关文档列表

        Returns:
            知识水平枚举
        """
        if not documents:
            logger.warning("没有提供相关文档，无法进行知识水平分析")
            return KnowledgeLevel.BEGINNER

        logger.info(f"开始LLM知识水平分析，documents: {documents}")

        try:
            # 构建分析提示词
            analysis_prompt = self._get_knowledge_level_analysis_prompt(query, documents)

            # 调用LLM进行分析
            analysis_result = await self.llm_service.generate(analysis_prompt)

            # 解析结果
            knowledge_level = self._parse_llm_response(analysis_result)

            logger.info(f"LLM知识水平分析完成: {knowledge_level.value}")
            return knowledge_level
        except Exception as e:
            logger.error(f"LLM知识水平推断失败: {e}", exc_info=True)
            return await self._fallback_knowledge_level_inference(query, documents)

    def _parse_llm_response(self, response: str) -> KnowledgeLevel:
        """
        解析LLM响应结果

        Args:
            response: LLM返回的响应

        Returns:
            解析出的知识水平
        """
        try:
            # 提取JSON部分 - 支持多种格式
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
                r'(\{[^{}]*"knowledge_level"[^{}]*\})'
            ]

            analysis_data = None
            for pattern in json_patterns:
                json_match = re.search(pattern, response, re.DOTALL)
                if json_match:
                    try:
                        analysis_data = json.loads(json_match.group(1))
                        break
                    except json.JSONDecodeError:
                        continue

            if analysis_data:
                level_str = analysis_data.get('knowledge_level', 'beginner').lower()
                confidence = analysis_data.get('confidence', 0.5)
                explanation = analysis_data.get('explanation', '')

                logger.info(f"LLM分析结果 - 水平: {level_str}, 置信度: {confidence}")
                logger.debug(f"分析说明: {explanation}")

                # 转换为枚举
                level_mapping = {
                    'advanced': KnowledgeLevel.ADVANCED,
                    'intermediate': KnowledgeLevel.INTERMEDIATE,
                    'beginner': KnowledgeLevel.BEGINNER
                }

                return level_mapping.get(level_str, KnowledgeLevel.BEGINNER)
            else:
                logger.warning("LLM返回结果中未找到有效JSON格式")
                # 尝试文本解析作为备选
                return self._parse_text_response(response)

        except Exception as e:
            logger.warning(f"解析LLM响应失败: {e}")
            return self._parse_text_response(response)

    def _parse_text_response(self, response: str) -> KnowledgeLevel:
        """
        文本格式响应解析备选方案

        Args:
            response: LLM响应文本

        Returns:
            解析出的知识水平
        """
        response_lower = response.lower()

        if 'advanced' in response_lower:
            return KnowledgeLevel.ADVANCED
        elif 'intermediate' in response_lower:
            return KnowledgeLevel.INTERMEDIATE
        else:
            return KnowledgeLevel.BEGINNER

    async def _fallback_knowledge_level_inference(self, query: str, documents: List[Document]) -> KnowledgeLevel:
        """
        备用的知识水平推断方法（基于规则）

        Args:
            query: 用户查询
            documents: 文档列表

        Returns:
            推断的知识水平
        """
        logger.info("使用备用知识水平推断方法")

        if not documents:
            return KnowledgeLevel.BEGINNER

        # 文档质量分析
        high_weight_docs = sum(1 for doc in documents if doc.metadata.get('weight', 0.8) >= 1.0)
        avg_weight = sum(doc.metadata.get('weight', 0.8) for doc in documents) / len(documents)
        has_code_examples = any('```' in doc.page_content for doc in documents)

        # 查询复杂度分析
        query_lower = query.lower()
        query_terms = query_lower.split()

        # 使用缓存的技术术语
        tech_terms = self._get_technical_terms()

        # 检查不同级别的术语
        basic_matches = sum(1 for term in tech_terms["basic"] if term in query_lower)
        intermediate_matches = sum(1 for term in tech_terms["intermediate"] if term in query_lower)
        advanced_matches = sum(1 for term in tech_terms["advanced"] if term in query_lower)

        # 综合评分
        score = 0.0

        # 文档质量权重
        score += min(high_weight_docs * 0.3, 0.9)
        score += min(avg_weight * 0.5, 0.5)

        # 查询复杂度权重
        if advanced_matches > 0:
            score += 0.8
        elif intermediate_matches > 0:
            score += 0.4
        elif basic_matches > 0:
            score -= 0.2

        # 特殊指标
        if has_code_examples:
            score += 0.2
        if len(query_terms) > 10:  # 长查询通常更复杂
            score += 0.1

        # 阈值判断
        if score >= 1.2:
            return KnowledgeLevel.ADVANCED
        elif score >= 0.6:
            return KnowledgeLevel.INTERMEDIATE
        else:
            return KnowledgeLevel.BEGINNER

    # 保留传统方法作为备选
    async def infer_knowledge_level_legacy(self, query: str, documents: List[Document]) -> KnowledgeLevel:
        """
        传统的基于词典的知识水平推断方法（保留作为备选）
        """
        logger.info("使用传统的基于词典的知识水平推断方法")
        return await self._fallback_knowledge_level_inference(query, documents)

    # 主要的公共接口
    async def infer_knowledge_level(self, query: str, documents: List[Document] = None) -> KnowledgeLevel:
        """
        智能推断用户知识水平（主要接口）

        Args:
            query: 用户查询
            documents: 相关文档列表，如果为None则会自动搜索

        Returns:
            推断的知识水平
        """
        if not query or not query.strip():
            logger.warning("查询为空，默认为初学者水平")
            return KnowledgeLevel.BEGINNER

        # 如果没有提供文档，则自动搜索
        if documents is None:
            logger.info("未提供文档，执行自动搜索")
            documents = await self.enhanced_search(query, k=5)

        # 使用LLM进行分析
        return await self.infer_knowledge_level_with_llm(query, documents)
