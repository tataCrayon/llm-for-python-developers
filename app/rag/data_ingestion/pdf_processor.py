from pathlib import Path
from typing import List, Optional
from langchain.docstore.document import Document
from app.config.logger import setup_logger
import PyPDF2
import uuid

logger = setup_logger(__name__)

class PDFProcessorConfig:
    """PDF 处理器配置类。"""
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

class PDFProcessor:
    """PDF 文件处理器。"""
    def __init__(self, config: Optional[PDFProcessorConfig] = None):
        """
        初始化 PDF 处理器。
        Args:
            config (Optional[PDFProcessorConfig]): 分块配置，默认为 1000 字符。
        """
        self.config = config or PDFProcessorConfig()
        logger.info(f"Initialized PDFProcessor with max_chunk_size: {self.config.max_chunk_size}")

    def process(self, file_path: Path) -> List[Document]:
        """
        处理 PDF 文件并生成 Document 对象。
        Args:
            file_path (Path): PDF 文件路径。
        Returns:
            List[Document]: 分块后的 Document 对象列表。
        """
        try:
            with file_path.open('rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

            # 按固定长度分块
            chunks = []
            for i in range(0, len(text), self.config.max_chunk_size):
                chunks.append(text[i:i + self.config.max_chunk_size])

            # 创建 Document 对象
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        'id': str(uuid.uuid4()),
                        'source': str(file_path),
                        'file_name': file_path.name,
                        'extension': file_path.suffix,
                        'chunk_index': i
                    }
                )
                for i, chunk in enumerate(chunks)
            ]

            logger.info(f"Processed {file_path}: {len(documents)} chunks")
            return documents
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            return []