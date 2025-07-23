from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document

from app.config.rag_prompt import RAGPromptConfig
from app.db.vector_store.chroma_store import ChromaVectorStore
from app.config.logger import setup_logger
import json

logger = setup_logger(__name__)

KNOWLEDGE_LEVEL_BEGINNER = 1
KNOWLEDGE_LEVEL_ADVANCED = 2

prompt_config = RAGPromptConfig()

def enhanced_search(query: str,
                    vector_store: ChromaVectorStore,
                    k: int = 5
                    ) -> List[Document]:
    """
    增强搜索函数，根据查询和向量数据库进行搜索，返回相关文档列表。

    :param query: 搜索查询字符串
    :param vector_store: 向量数据库实例
    :param k: 返回的文档数量，默认为 5
    :return: 相关文档列表
    """
    logger.info(f"Performing enhanced search for query: {query}")
    try:
        results = vector_store.search_documents(query, k=k)
        ranked_results = []
        for doc in results:
            keywords = doc.metadata.get('keywords', '').split(',')
            weight = doc.metadata.get('weight', 0.8)
            context = json.loads(doc.metadata.get('context', '{}'))
            relevance_score = weight # 初始分数基于权重
            if any(keyword.lower() in query.lower() for keyword in keywords):
                relevance_score += 0.2 # 关键词匹配加分
            ranked_results.append((doc, relevance_score))
        # 按分数排序
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_results[:k]]
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}", exc_info=True)
        return []


def infer_knowledge_level(documents: List[Document]) -> str:
    """
    知识水平推断。
    :param documents: 相关文档列表
    :return: 知识水平字整数level
    """
    if not documents:
        return KNOWLEDGE_LEVEL_BEGINNER
    technical_terms = ["embeddings", "cosine similarity", "vector database", "RAG"]
    has_technical_content = False
    high_weight_count = sum(1 for doc in documents if doc.metadata.get('weight', 0.8) >= 1.0)
    for doc in documents:
        if any(term in doc.page_content.lower() for term in technical_terms):
            has_technical_content = True
            break
    if has_technical_content or high_weight_count >= 2:
        return KNOWLEDGE_LEVEL_ADVANCED
    return KNOWLEDGE_LEVEL_BEGINNER


def generate_answer(query: str, documents: List[Document], knowledge_level: str) -> Dict:
    logger.info(f"Generating answer for query: {query}, knowledge level: {knowledge_level}")
    try:
        context = "\n".join([
            f"Source: {doc.metadata['source']}, Chunk {doc.metadata['chunk_index']}\n"
            f"Keywords: {doc.metadata.get('keywords', '')}\n"
            f"Content: {doc.page_content[:200]}..."
            for doc in documents
        ])
        prompt = f"""
        {prompt_config.rag_prompt_content}
        **Retrieved Documents**:
        {context}
        **Knowledge Level**: {knowledge_level}
        **Query**: {query}
        """
        # 替换为实际 AI 模型调用（例如 Grok 3）
        response = {
            "answer": f"Based on your notes, {'here is a detailed explanation' if knowledge_level == 'advanced' else 'here is a simple explanation'}: ...",
            "sources": [f"{doc.metadata['source']} (chunk {doc.metadata['chunk_index']})" for doc in documents],
            "additional_notes": "Explore related topics like embeddings or RAG."
        }
        logger.info(f"Generated answer: {response['answer'][:50]}...")
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return {"answer": "", "sources": [], "additional_notes": ""}


def rag_pipeline(query: str, vector_store: ChromaVectorStore) -> Dict:
    """
    RAG 流程：检索、推断知识水平、生成回答。
    Args:
        query (str): 用户查询。
        vector_store (ChromaVectorStore): Chroma 向量存储实例。
    Returns:
        Dict: 包含答案、来源和附加笔记。
    """
    logger.info(f"Starting RAG pipeline for query: {query}")
    documents = enhanced_search(query, vector_store)
    knowledge_level = infer_knowledge_level(documents)
    response = generate_answer(query, documents, knowledge_level)
    return response


if __name__ == "__main__":
    vector_store = ChromaVectorStore()
    query = "What is a vector database?"
    result = rag_pipeline(query, vector_store)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Additional Notes: {result['additional_notes']}")
