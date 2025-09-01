import asyncio
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_recall, context_precision

from app.config.deepseek_llm import deep_seek_chat_model
from app.config.embedding_model import EmbeddingModelComponent
from app.config.logger import setup_logger
from app.db.vector_store.chroma_store import ChromaVectorStore
from app.rag.data_ingestion.md_processor import MDProcessor, MDProcessorConfig
from app.rag.data_ingestion.ragas_dataset_generator import RagasDatasetGenerator

logger = setup_logger(__name__)

# --- 1. 配置评估参数 ---
# 选择你要评估的Markdown文件
MD_FILE_PATH = Path("../../data/raw_data/异步通信P1—消息队列基本概念.md")
# 选择你要评估的分块策略
CHUNK_STRATEGY_TO_TEST = "recursive"
# 生成的评估问题数量
EVAL_TEST_SIZE = 20
# 检索时返回的Top-K个结果
RETRIEVER_TOP_K = 3
# 是否使用强模型生成测试集（强烈推荐）
USE_STRONG_MODEL = True


async def run_evaluation():
    """
    执行端到端的Chunking策略评估
    """

    logger.info(f"--- 开始评估，策略: {CHUNK_STRATEGY_TO_TEST}, 文件: {MD_FILE_PATH.name} ---")

    # --- 2. 使用你的处理器进行分块 ---
    config = MDProcessorConfig(chunk_strategy=CHUNK_STRATEGY_TO_TEST)
    processor = MDProcessor(config=config)
    documents = processor.process(MD_FILE_PATH)
    if not documents:
        logger.info("错误：未能从文件中生成任何chunks。")
        return

    logger.info(f"步骤1: 分块完成，生成 {len(documents)} 个 chunks。")

    # --- 3. 生成评估数据集（使用强模型） ---
    logger.info(f"步骤2: 开始生成评估数据集（使用{'强模型' if USE_STRONG_MODEL else '小模型'}）...")
    generator = RagasDatasetGenerator()
    eval_df = generator.generate_ragas_dataset_sync(
        documents=documents,
        test_size=EVAL_TEST_SIZE,
        use_strong_model=USE_STRONG_MODEL  # 使用强模型参数
    )

    logger.info(f"步骤3: Ragas评估数据集生成完成，包含 {len(eval_df)} 个问题。")

    # 验证生成的数据集格式
    required_columns = ["user_input", "reference_contexts"]
    missing_columns = [col for col in required_columns if col not in eval_df.columns]
    if missing_columns:
        logger.info(f"警告：生成的数据集缺少必要的列: {missing_columns}")
        logger.info(f"实际的列: {eval_df.columns.tolist()}")
        return

    logger.info(f"数据集包含的列: {eval_df.columns.tolist()}")
    # --- 4. 构建向量数据库和检索器 ---
    # 确保使用与Ragas生成问题时相同的Embedding模型
    embeddings = EmbeddingModelComponent.get_default()
    # 为本次评估创建一个临时的collection name，避免冲突
    collection_name = f"eval-simple-{CHUNK_STRATEGY_TO_TEST}"
    # 创建新的ChromaVectorStore实例，使用特定的collection name
    vector_store = ChromaVectorStore(
        embeddings=embeddings,
        collection_name=collection_name
    )

    vector_store.add_documents(documents)

    logger.info(f"步骤4: 向量数据库构建完成，Collection: {collection_name}")

    # --- 5. 准备评估数据并执行检索 ---
    questions = eval_df["user_input"].tolist()
    ground_truth_contexts = eval_df["reference_contexts"].tolist()
    references = eval_df["reference"].tolist()

    logger.info(f"准备评估数据:")
    logger.info(f"  - 问题数量: {len(questions)}")
    logger.info(f"  - 参考contexts数量: {len(ground_truth_contexts)}")
    logger.info(f"  - 参考答案数量: {len(references)}")

    retrieved_contexts = []
    for i, q in enumerate(questions):
        try:
            # 修复1: 正确调用search_documents方法
            retrieved_docs = vector_store.search_documents(query=q, k=RETRIEVER_TOP_K)
            retrieved_contexts.append([doc.page_content for doc in retrieved_docs])

            # 记录检索进度
            if (i + 1) % 5 == 0:
                logger.info(f"  检索进度: {i + 1}/{len(questions)}")
        except Exception as e:
            logger.error(f"检索问题 '{q[:50]}...' 时出错: {e}")
            retrieved_contexts.append([])  # 添加空结果

    logger.info("步骤5: 已对所有评估问题执行检索。")

    # --- 6. 使用Ragas进行评估 ---
    try:
        # 准备评估数据
        data = {
            "question": questions,
            "contexts": retrieved_contexts,
            "ground_truth_contexts": ground_truth_contexts,
            "reference": references  # 使用现有的reference列
        }

        # 验证数据长度一致性
        lengths = {k: len(v) for k, v in data.items()}
        if len(set(lengths.values())) > 1:
            logger.info(f"错误: 数据列长度不一致: {lengths}")
            return

        logger.info(f"评估数据准备完成，各列长度: {lengths}")

        eval_dataset = Dataset.from_dict(data)

        logger.info("步骤6: 开始Ragas评估...")

        evaluator_llm = LangchainLLMWrapper(deep_seek_chat_model())
        # 执行评估
        score = evaluate(
            eval_dataset,
            metrics=[context_recall, context_precision],
            # context_recall, context_precision 评估分块
            # faithfulness, FactualCorrectness() 评估答案
            llm=evaluator_llm
        )

        logger.info(f"步骤6: Ragas评估完成:{score}")

        # --- 7. 打印结果 ---
        results_df = score.to_pandas()
        avg_recall = results_df['context_recall'].mean()
        avg_precision = results_df['context_precision'].mean()

        logger.info("\n" + "=" * 50)
        logger.info("评估结果")
        logger.info("=" * 50)
        logger.info(f"分块策略: {CHUNK_STRATEGY_TO_TEST}")
        logger.info(f"文件: {MD_FILE_PATH.name}")
        logger.info(f"生成的测试问题数: {len(questions)}")
        logger.info(f"检索Top-K: {RETRIEVER_TOP_K}")
        logger.info("-" * 50)
        logger.info(f"平均 Context Recall: {avg_recall:.4f}")
        logger.info(f"平均 Context Precision: {avg_precision:.4f}")
        logger.info("-" * 50)

        # 显示一些具体的例子
        logger.info("\n前3个问题的详细分数:")
        for i in range(min(3, len(results_df))):
            logger.info(f"\nQ{i + 1}: {questions[i][:80]}...")
            logger.info(f"  Recall: {results_df.iloc[i]['context_recall']:.3f}")
            logger.info(f"  Precision: {results_df.iloc[i]['context_precision']:.3f}")
            logger.info(f"  检索到的contexts数量: {len(retrieved_contexts[i])}")
            logger.info(
                f"  真实contexts数量: {len(ground_truth_contexts[i]) if isinstance(ground_truth_contexts[i], list) else 1}")

        # 保存详细结果
        output_file = f"eval_results_{CHUNK_STRATEGY_TO_TEST}_{MD_FILE_PATH.stem}.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"\n详细结果已保存至: {output_file}")

        # 额外的统计信息
        logger.info(f"\n统计信息:")
        logger.info(f"  成功检索的问题数: {sum(1 for ctx in retrieved_contexts if ctx)}")
        logger.info(f"  检索失败的问题数: {sum(1 for ctx in retrieved_contexts if not ctx)}")
        logger.info(
            f"  平均检索contexts数: {sum(len(ctx) for ctx in retrieved_contexts) / len(retrieved_contexts):.1f}")

    except Exception as e:
        logger.error(f"Ragas评估过程中出错: {e}")
        import traceback
        traceback.logger.info_exc()


if __name__ == "__main__":
    asyncio.run(run_evaluation())
