from pathlib import Path

from app.rag.data_ingestion.knowledge_base_loader import KnowledgeBaseLoader
from app.config.logger import setup_logger

# 初始化日志
logger = setup_logger(__name__)


def test_load_file():
    try:
        file_path = Path(r"G:\Note\职业发展\AI\RAG\RAG轻松通-P3：向量数据库.md")
        logger.info(f"开始加载文件: {file_path}")
        loader = KnowledgeBaseLoader()
        ids = loader.load(file_path)
        logger.info(f"文件加载完成: {file_path}, IDs: {ids}")
    except Exception as e:
        logger.error(f"加载文件时出错: {e}", exc_info=True)
        raise

def test_load_directory():
    try:
        dir_path = Path(r"G:\Note\职业发展")
        logger.info(f"开始加载目录: {dir_path}")
        loader = KnowledgeBaseLoader()
        ids = loader.load(dir_path)
        logger.info(f"目录加载完成: {dir_path}, IDs: {ids}")
    except Exception as e:
        logger.error(f"加载目录时出错: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # test_load_file()
    test_load_directory()
