"""
KeywordExtractor 简化测试模块

本模块包含对 KeywordExtractor 类的简化测试，
仅验证关键词提取功能的基本效果。
"""

import logging
import os
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 将项目根目录添加到 Python 路径中，以便导入应用模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.utils.keyword_extractor import KeywordExtractor


def test_extract_keywords_basic():
    """测试基本的关键词提取功能"""
    logger.info("开始执行基本关键词提取测试")

    # 创建 KeywordExtractor 实例
    extractor = KeywordExtractor()
    logger.info("KeywordExtractor 实例已创建")

    # 测试文本
    test_text = "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
    logger.info("测试文本已设置")

    # 执行关键词提取
    keywords = extractor.extract_keywords(test_text, 5)
    logger.info(f"关键词提取完成，结果: {keywords}")

    # 验证结果不为空
    assert len(keywords) > 0
    logger.info("基本关键词提取测试通过")


def test_extract_keywords_with_whitespace():
    """测试包含空格的关键词提取"""
    logger.info("开始执行带空格的关键词提取测试")

    # 创建 KeywordExtractor 实例
    extractor = KeywordExtractor()
    logger.info("KeywordExtractor 实例已创建")

    # 测试文本
    test_text = "人工智能是计算机科学的一个分支。"
    logger.info("测试文本已设置")

    # 执行关键词提取
    keywords = extractor.extract_keywords(test_text, 5)
    logger.info(f"关键词提取完成，结果: {keywords}")

    # 验证结果不为空
    assert len(keywords) > 0
    logger.info("带空格的关键词提取测试通过")


if __name__ == "__main__":
    test_extract_keywords_basic()
    test_extract_keywords_with_whitespace()
    logger.info("所有测试已通过！")
