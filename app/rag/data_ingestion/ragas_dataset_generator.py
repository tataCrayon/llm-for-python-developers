import asyncio
import os
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter

from app.config.deepseek_llm import deep_seek_chat_model
from app.config.embedding_model import EmbeddingModelComponent
from app.config.logger import setup_logger
from scripts.evaluation.user_def import user_personas

logger = setup_logger(__name__)


class RagasDatasetGenerator:
    """
    Ragas数据集生成器类，用于生成用于评估RAG系统的测试数据集。
    """

    def __init__(self,
                 llm: LangchainLLMWrapper | None = None,
                 embedding_model: LangchainEmbeddingsWrapper | None = None):
        """
        初始化RagasDatasetGenerator实例。

        Args:
            llm (Optional[LangchainLLMWrapper]): 用于生成数据集的语言模型。
            embedding_model (Optional[LangchainEmbeddingsWrapper]): 用于生成数据集的嵌入模型。
        """
        self.llm_wrapper = llm or LangchainLLMWrapper(deep_seek_chat_model())
        self.embedding_wrapper = embedding_model or LangchainEmbeddingsWrapper(EmbeddingModelComponent.get_default())

    def generate_ragas_dataset_sync(self,
                                    documents: List[Document],
                                    test_size: int = 20,
                                    output_path: str = "ragas_eval_dataset.jsonl",
                                    use_strong_model: bool = True) -> pd.DataFrame:
        """
        同步方法：使用Ragas的TestsetGenerator为Chunking评估生成一个测试集

        Args:
            documents (List[Document]): 用于生成测试集的文档列表 (即您的Chunks)
            test_size (int): 要生成的测试样本数量
            output_path (str): 保存生成的数据集的文件路径
            use_strong_model (bool): 是否使用强模型（推荐用于生成高质量测试集）

        Returns:
            pd.DataFrame: 生成的测试集
        """
        # 使用asyncio.run来运行异步方法
        return asyncio.run(self.generate_ragas_dataset(documents, test_size, output_path, use_strong_model))

    async def generate_ragas_dataset(self,
                                     documents: List[Document],
                                     test_size: int = 20,
                                     output_path: str = "ragas_eval_dataset.jsonl",
                                     use_strong_model: bool = True) -> pd.DataFrame:

        """
        使用Ragas的TestsetGenerator为Chunking评估生成一个测试集

        Args:
            documents (List[Document]): 用于生成测试集的文档列表 (即您的Chunks)
            test_size (int): 要生成的测试样本数量
            output_path (str): 保存生成的数据集的文件路径
            use_strong_model (bool): 是否使用强模型（推荐用于生成高质量测试集）

        Returns:
            pd.DataFrame: 生成的测试集
        """

        if not documents:
            logger.warning("文档列表为空，无法生成测试集。")
            return pd.DataFrame()

        query_dist = {
            "simple": 0.5,
            "reasoning": 0.25,
            "multi_context": 0.25
        }

        transforms = [HeadlineSplitter(), NERExtractor(llm=self.llm_wrapper)]

        if user_personas:
            logger.info(f"使用 {len(user_personas)} 个自定义角色...")
            # transforms = Transforms(personas=user_personas)
            # 当使用 persona 时，test_size 通常由 persona 内的 questions_per_persona 总和决定
            # 为简单起见，我们这里依然使用外部的 test_size 参数

        generator = TestsetGenerator(
            llm=self.llm_wrapper,
            embedding_model=self.embedding_wrapper,
            persona_list=user_personas,
        )

        from ragas.testset.synthesizers.single_hop.specific import (
            SingleHopSpecificQuerySynthesizer,
        )

        distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.llm_wrapper), 1.0),
        ]

        for query, _ in distribution:
            prompts = await query.adapt_prompts("chinese", llm=self.llm_wrapper)
            query.set_prompts(**prompts)

        logger.info(f"开始生成测试集，样本数量: {test_size}...")

        try:
            # 尝试生成测试集
            test_data_set = generator.generate_with_langchain_docs(
                documents=documents,
                # 如果使用小模型，减少批次大小以降低复杂度
                testset_size=1 if not use_strong_model else 5,
                query_distribution=distribution,
                transforms=transforms,
                with_debugging_logs=True,  # 开启调试日志
                raise_exceptions=True,
            )
        except Exception as gen_error:
            logger.error(f"生成失败: {gen_error}")

        # 转换为DataFrame
        df = test_data_set.to_pandas()

        # 检查并记录数据集的列信息
        logger.info(f"生成的数据集列名: {list(df.columns)}")

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存为jsonl格式
        df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        logger.info(f"Ragas测试集生成完毕，共生成 {len(df)} 个样本，保存至 {output_path}")

        # 验证数据集是否包含评估所需的必要列
        required_columns = ['question', 'contexts']
        missing_columns = [col for col in required_columns if col not in df.columns]

        # 检查是否有替代列名
        alternative_columns = {
            'question': ['user_input'],
            'contexts': ['reference_contexts']
        }

        for col in missing_columns[:]:
            for alt_col in alternative_columns.get(col, []):
                if alt_col in df.columns:
                    missing_columns.remove(col)
                    break

        if missing_columns:
            logger.warning(f"警告：生成的数据集缺少必要的列: {missing_columns}")
            logger.info(f"实际的列: {list(df.columns)}")
        else:
            logger.info("数据集包含所有必要列，可用于Ragas评估。")

        # 打印前几个样本以供检查
        if len(df) > 0:
            logger.info(f"生成的问题示例：")
            # 使用正确的列名访问问题内容
            question_column = 'user_input' if 'user_input' in df.columns else 'question'
            for i, row in df.head(3).iterrows():
                question_text = row.get(question_column, 'N/A')
                if isinstance(question_text, list) and len(question_text) > 0:
                    # 如果是列表，取第一个元素
                    question_text = question_text[0]
                logger.info(f"  Q{i + 1}: {str(question_text)[:100]}...")

        logger.info(f"生成的数据集列名: {list(df.columns)}")

        return df
