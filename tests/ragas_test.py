import asyncio

from datasets import load_dataset
from ragas import EvaluationDataset
from ragas import SingleTurnSample
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic

from app.config.embedding_model import EmbeddingModelComponent
from app.config.small_model import KeywordModelComponent

evaluator_llm = LangchainLLMWrapper(KeywordModelComponent.get_default())
evaluator_embeddings = LangchainEmbeddingsWrapper(EmbeddingModelComponent.get_default())

test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
}

test_sample = SingleTurnSample(**test_data)

metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate.")
test_data = SingleTurnSample(**test_data)

result = asyncio.run(metric.single_turn_ascore(test_sample))

print(result)

# 数据集评估

eval_dataset = load_dataset("explodinggradients/earning_report_summary", split="train")
eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
print("Features in dataset:", eval_dataset.features())
print("Total samples in dataset:", len(eval_dataset))

results = evaluate(eval_dataset, metrics=[metric])

print(results)
print(results.to_pandas())
