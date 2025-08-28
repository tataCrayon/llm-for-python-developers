from app.config.logger import setup_logger
from app.services.query_analysis_service import QueryAnalysisService

logger = setup_logger(__name__)

if __name__ == '__main__':

    query_analyzer = QueryAnalysisService()

    # 2. 定义一组测试查询
    queries_to_test = [
        "你好，今天天气怎么样？",  # 应该返回 0
        # "Transformer模型的核心是什么？",  # 应该返回 0 (通用知识)
        # "根据我的知识库，总结一下我关于Agentic RAG架构的理解。",  # 应该返回 1
        # "怎么评估RAG系统的性能？",  # 应该返回 1 (深度专业问题)
        # "帮我写一封邮件，邀请我的同事参加技术分享会。",  # 应该返回 0
        # "我上次记录的关于ChromaDB的几个关键参数是什么来着？",  # 应该返回 1
    ]

    # 3. 运行分析并打印结果
    for q in queries_to_test:
        route, reason = query_analyzer.analyze_query(q)
        decision = "✅ 需要走RAG" if route == 1 else "❌ 直接问LLM"
        logger.info(f"查询: '{q}'")
        logger.info(f"决策: {decision} (代码: {route})")
        logger.info(f"理由: {reason}")
        logger.info("-" * 20)
