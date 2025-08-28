import json
import os
import time
from datetime import datetime

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config.deepseek_llm import deep_seek_chat_model, deep_seek_r1_model
from app.config.logger import setup_logger
# 从我们创建的文件中导入数据集
from evaluation_dataset import EVALUATION_DATASET

logger = setup_logger(__name__)

# --- 1. 配置语言模型 ---
# 用于执行Prompt的模型
# 建议使用性价比高的模型，如 deepseek-chat
generator_llm = deep_seek_chat_model()

# 用于评估的“裁判”模型，使用更强大的模型，以保证评估质量
judge_llm = deep_seek_r1_model()

# --- 2. 定义您要测试的Multi-query Prompt ---
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

multi_query_prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT_TEMPLATE)


# --- 3. 定义“裁判LLM”的Prompt和输出结构 ---

# Pydantic模型定义了裁判打分的输出结构，确保输出稳定
class SubQueryEvaluation(BaseModel):
    relevance_score: int = Field(description="评估生成的问题与原始问题的相关性，1-5分")
    coverage_score: int = Field(description="评估生成的问题是否全面覆盖了回答原始问题所需的关键信息点，1-5分")
    atomicity_score: int = Field(description="评估生成的问题是否足够具体和原子化，适合向量检索，1-5分")
    reasoning: str = Field(description="对打分结果的简要说明和改进建议")


JUDGE_PROMPT_TEMPLATE = """
你是一位严谨的AI评测专家。你的任务是评估一个AI模型生成的“子问题列表”的质量。

# 背景
- **原始用户问题**: 这是用户最开始提出的问题。
- **理想的子问题 (Ground Truth)**: 这是专家定义的、高质量的子问题分解，作为参考标准。
- **生成的子问题**: 这是AI模型根据“原始用户问题”实际生成的子问题列表。

# 评估维度 (请在1-5分之间打分):
1.  **相关性 (Relevance)**: “生成的子问题”是否都与“原始用户问题”的核心意图紧密相关？（5分表示高度相关，1分表示完全无关）
2.  **覆盖度 (Coverage)**: “生成的子问题”集合是否覆盖了回答“原始用户问题”所需的全部关键信息点？可以参考“理想的子问题”来判断覆盖范围。（5分表示完全覆盖，1分表示严重缺失关键信息）
3.  **原子性 (Atomicity)**: 每个“生成的子问题”是否足够具体、独立，适合作为单一的检索单元？（5分表示每个问题都是清晰、原子的，1分表示问题模糊、宽泛或包含多个问题）

# 你的任务
请根据上述评估维度，对【生成的子问题】进行打分，并提供简要的【评分理由】。请严格按照以下JSON格式输出，不要添加任何额外的文本或解释：


# 输入数据
- **原始用户问题**: {original_question}
- **理想的子问题**: {ideal_sub_queries}
- **生成的子问题**: {generated_sub_queries}

# 输出JSON
{{
  "relevance_score": 5,
  "coverage_score": 5,
  "atomicity_score": 5,
  "reasoning": "对打分结果的简要说明和改进建议（如果需要）"
}}
"""

judge_prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT_TEMPLATE)
judge_parser = JsonOutputParser(pydantic_object=SubQueryEvaluation)
judge_chain = judge_prompt | judge_llm | judge_parser


# --- 4. 运行评估流程 ---

def run_evaluation():
    """
    主评估函数
    """
    results = []

    # 创建保存结果的目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"evaluation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 定义生成子问题的链
    generation_chain = multi_query_prompt | generator_llm

    # 评估进度跟踪
    total_questions = len(EVALUATION_DATASET)
    logger.info(f"开始评估，共 {total_questions} 个问题")

    for idx, item in enumerate(EVALUATION_DATASET, 1):
        start_time = time.time()
        logger.info(f"--- 正在评估 Question ID: {item['question_id']} ({idx}/{total_questions}) ---")

        # 4a. 生成子问题
        response = generation_chain.invoke({"original_question": item["original_question"]})
        try:
            # LLM的输出是字符串，需要解析成JSON
            generated_queries = json.loads(response.content)
            logger.debug(f"生成的子问题: {generated_queries}")
        except json.JSONDecodeError:
            logger.warning(f"  [错误] 无法解析LLM输出: {response.content}")
            generated_queries = []

        # 4b. 调用裁判LLM进行评估
        try:
            judge_response = judge_chain.invoke({
                "original_question": item["original_question"],
                "ideal_sub_queries": str(item["ideal_sub_queries"]),
                "generated_sub_queries": str(generated_queries)
            })
        except Exception as e:
            logger.warning(f"  [错误] 调用裁判LLM失败: {e}")
            # 提供默认的评估结果
            judge_response = {
                "relevance_score": 0,
                "coverage_score": 0,
                "atomicity_score": 0,
                "reasoning": f"调用裁判LLM失败: {e}"
            }

        # 确保裁判响应包含所有必要的字段
        # 如果某些字段缺失，使用默认值填充
        required_fields = ['relevance_score', 'coverage_score', 'atomicity_score', 'reasoning']
        for field in required_fields:
            if field not in judge_response:
                if field == 'reasoning':
                    judge_response[field] = "字段缺失"
                else:
                    judge_response[field] = 0  # 默认分数为0

        # 记录评估耗时
        evaluation_time = time.time() - start_time

        # 记录结果
        result_record = {
            "question_id": item["question_id"],
            "original_question": item["original_question"],
            "ideal_sub_queries": item["ideal_sub_queries"],
            "generated_sub_queries": generated_queries,
            "evaluation_time_seconds": evaluation_time,
            **judge_response  # 将裁判的打分和理由合并进来
        }
        results.append(result_record)

        # 保存中间结果
        intermediate_results_path = os.path.join(results_dir, f"intermediate_results_{item['question_id']}.json")
        with open(intermediate_results_path, 'w', encoding='utf-8') as f:
            json.dump(result_record, f, ensure_ascii=False, indent=2)

        logger.info(f"  完成评估 Question ID: {item['question_id']}, 耗时: {evaluation_time:.2f}秒")
        logger.info(f"  相关性得分: {judge_response.get('relevance_score', 'N/A')}")
        logger.info(f"  覆盖度得分: {judge_response.get('coverage_score', 'N/A')}")
        logger.info(f"  原子性得分: {judge_response.get('atomicity_score', 'N/A')}")
        logger.info(f"  评分理由: {judge_response.get('reasoning', 'N/A')}")

    return pd.DataFrame(results)


def analyze_results(results_df) -> pd.DataFrame:
    """
    分析评估结果
    """
    logger.info("\n\n--- 评估结果详细分析 ---")

    # 基本统计信息
    logger.info("基本统计信息:")
    logger.info(f"  总问题数: {len(results_df)}")
    logger.info(f"  平均评估时间: {results_df['evaluation_time_seconds'].mean():.2f}秒")

    # 分数分布
    logger.info("\n分数分布:")
    for score_type in ['relevance_score', 'coverage_score', 'atomicity_score']:
        logger.info(f"  {score_type}:")
        logger.info(f"    平均分: {results_df[score_type].mean():.2f}")
        logger.info(f"    最高分: {results_df[score_type].max()}")
        logger.info(f"    最低分: {results_df[score_type].min()}")
        logger.info(f"    标准差: {results_df[score_type].std():.2f}")

    # 分数分布直方图
    logger.info("\n分数分布直方图:")
    for score_type in ['relevance_score', 'coverage_score', 'atomicity_score']:
        logger.info(f"  {score_type}分布:")
        value_counts = results_df[score_type].value_counts().sort_index()
        for score in range(1, 6):
            count = value_counts.get(score, 0)
            percentage = (count / len(results_df)) * 100
            logger.info(f"    {score}分: {count}个 ({percentage:.1f}%)")

    # 优秀和较差案例
    logger.info("\n优秀案例 (各维度得分>=4):")
    excellent_cases = results_df[
        (results_df['relevance_score'] >= 4) &
        (results_df['coverage_score'] >= 4) &
        (results_df['atomicity_score'] >= 4)
        ]
    for _, row in excellent_cases.iterrows():
        logger.info(f"  Question ID: {row['question_id']}")
        logger.info(f"    原始问题: {row['original_question']}")
        logger.info(f"    生成的子问题: {row['generated_sub_queries']}")
        logger.info(f"    评分理由: {row['reasoning']}")

    logger.info("\n需要改进的案例 (任一维度得分<=2):")
    poor_cases = results_df[
        (results_df['relevance_score'] <= 2) |
        (results_df['coverage_score'] <= 2) |
        (results_df['atomicity_score'] <= 2)
        ]
    for _, row in poor_cases.iterrows():
        logger.info(f"  Question ID: {row['question_id']}")
        logger.info(f"    原始问题: {row['original_question']}")
        logger.info(f"    生成的子问题: {row['generated_sub_queries']}")
        logger.info(f"    评分理由: {row['reasoning']}")


def save_results(results_df, timestamp) -> None:
    """
    保存评估结果
    """
    results_dir = f"evaluation_results_{timestamp}"

    # 保存完整的评估结果
    full_results_path = os.path.join(results_dir, "full_evaluation_results.csv")
    results_df.to_csv(full_results_path, index=False, encoding='utf-8')

    # 保存摘要结果
    summary_results_path = os.path.join(results_dir, "summary_evaluation_results.csv")
    summary_columns = ['question_id', 'relevance_score', 'coverage_score', 'atomicity_score', 'reasoning',
                       'evaluation_time_seconds']
    results_df[summary_columns].to_csv(summary_results_path, index=False, encoding='utf-8')

    logger.info(f"\n评估结果已保存到目录: {results_dir}")
    logger.info(f"  完整结果: {full_results_path}")
    logger.info(f"  摘要结果: {summary_results_path}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = run_evaluation()

    # 分析结果
    analyze_results(results_df)

    # 保存结果
    save_results(results_df, timestamp)

    logger.info("\n--- 评估完成 ---")
