"""
KeywordExtractor 简化测试模块

本模块包含对 KeywordExtractor 类的简化测试，
仅验证关键词提取功能的基本效果。
"""

import logging
import os
import sys
from unittest.mock import patch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 将项目根目录添加到 Python 路径中，以便导入应用模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 模拟 langchain_openai 模块，因为在测试环境中可能不可用
try:
    from app.utils.keyword_extractor import KeywordExtractor
except ImportError:
    # 为测试目的创建一个模拟版本
    import sys
    from unittest.mock import MagicMock

    # 创建模拟模块
    mock_langchain_openai = MagicMock()
    mock_langchain_openai.ChatOpenAI = MagicMock()

    # 将其添加到 sys.modules 中，使导入正常工作
    sys.modules['langchain_openai'] = mock_langchain_openai

    # 现在导入 KeywordExtractor
    from app.utils.keyword_extractor import KeywordExtractor


def test_extract_keywords_basic():
    """测试基本的关键词提取功能"""
    logger.info("开始执行基本关键词提取测试")
    # 设置环境变量
    with patch.dict(os.environ, {
        'LOCAL_MODEL_SERVE_BASE_URL': 'http://localhost:8000/v1',
        'LOCAL_MODEL_SERVE_MODEL_PATH': './Qwen2.5-1.5B-Instruct/Qwen/Qwen2___5-1___5B-Instruct'
    }):
        logger.info("环境变量已设置")
        # 创建 KeywordExtractor 实例
        extractor = KeywordExtractor()
        logger.info("KeywordExtractor 实例已创建")

        # 模拟 LLM 响应
        mock_response = MagicMock()
        mock_response.content = '人工智能, 机器学习, 深度学习, 神经网络, 自然语言处理'
        extractor.llm.invoke = MagicMock(return_value=mock_response)
        logger.info("LLM 响应已模拟")

        # 测试文本
        test_text = "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
        logger.info("测试文本已设置")

        # 执行关键词提取
        keywords = extractor.extract_keywords(test_text, 5)
        logger.info(f"关键词提取完成，结果: {keywords}")

        # 验证结果
        expected_keywords = ['人工智能', '机器学习', '深度学习', '神经网络', '自然语言处理']
        logger.info(f"预期结果: {expected_keywords}")
        assert keywords == expected_keywords
        logger.info("基本关键词提取测试通过")


def test_extract_keywords_with_whitespace():
    """测试包含空格的关键词提取"""
    logger.info("开始执行带空格的关键词提取测试")
    # 设置环境变量
    with patch.dict(os.environ, {
        'LOCAL_MODEL_SERVE_BASE_URL': 'http://localhost:8000/v1',
        'LOCAL_MODEL_SERVE_MODEL_PATH': './Qwen2.5-1.5B-Instruct/Qwen/Qwen2___5-1___5B-Instruct'
    }):
        logger.info("环境变量已设置")
        # 创建 KeywordExtractor 实例
        extractor = KeywordExtractor()
        logger.info("KeywordExtractor 实例已创建")

        # 模拟包含空格的 LLM 响应
        mock_response = MagicMock()
        mock_response.content = ' 人工智能 , 机器学习 , 深度学习 , 神经网络 , 自然语言处理 '
        extractor.llm.invoke = MagicMock(return_value=mock_response)
        logger.info("带空格的 LLM 响应已模拟")

        # 测试文本
        test_text = "人工智能是计算机科学的一个分支。"
        logger.info("测试文本已设置")

        # 执行关键词提取
        keywords = extractor.extract_keywords(test_text, 5)
        logger.info(f"关键词提取完成，结果: {keywords}")

        # 验证结果（应去除空格）
        expected_keywords = ['人工智能', '机器学习', '深度学习', '神经网络', '自然语言处理']
        logger.info(f"预期结果: {expected_keywords}")
        assert keywords == expected_keywords
        logger.info("带空格的关键词提取测试通过")


if __name__ == "__main__":
    test_extract_keywords_basic()
    test_extract_keywords_with_whitespace()
    print("所有测试已通过！")
