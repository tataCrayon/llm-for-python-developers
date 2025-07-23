from pathlib import Path
from typing import List, Optional, Dict
from app.db.vector_store.chroma_store import ChromaVectorStore
from app.rag.data_ingestion.md_processor import MDProcessor, MDProcessorConfig
from app.rag.data_ingestion.pdf_processor import PDFProcessor, PDFProcessorConfig
from app.config.logger import setup_logger
from dotenv import load_dotenv
import os

logger = setup_logger(__name__)
load_dotenv()

MAX_BATCH_SIZE = 3000 # 最大支持5461

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR")

if KNOWLEDGE_BASE_DIR:
    # 检查路径是否存在
    path = Path(KNOWLEDGE_BASE_DIR)
    if not path.exists():
        logger.warning(f"指定的知识库目录 {KNOWLEDGE_BASE_DIR} 不存在")
    elif not path.is_dir():
        logger.warning(f"{KNOWLEDGE_BASE_DIR} 不是一个有效的目录")
else:
    logger.warning("未设置 KNOWLEDGE_BASE_DIR 环境变量")


class KnowledgeBaseLoader:
    """
    知识库加载器，扫描目录并加载文件到向量数据库。
    """

    def __init__(
            self,
            vector_store: ChromaVectorStore | None = None,
            directory: Path | None = None,
            processor_configs: Dict[str, object] | None = None
    ):
        """
        初始化知识库加载器。
        Args:
            vector_store (ChromaVectorStore): 向量存储实例。
            directory (Path): 要扫描的目录，默认为当前目录。
            processor_configs (Optional[Dict[str, object]]): 文件类型对应的处理器配置。
        """
        logger.info("Start to initialize KnowledgeBaseLoader")
        self.vector_store = vector_store if vector_store is not None else ChromaVectorStore()
        self.directory = directory if directory is not None else Path(KNOWLEDGE_BASE_DIR)
        self.processor_configs = processor_configs or {}

        # 处理器注册表
        self.processors = {
            '.md': MDProcessor(self.processor_configs.get('.md', MDProcessorConfig())),
            '.pdf': PDFProcessor(self.processor_configs.get('.pdf', PDFProcessorConfig()))
        }
        logger.info(f"Initialized KnowledgeBaseLoader, default directory: {self.directory}")
        logger.info(f"Initialized KnowledgeBaseLoader, default processors: {self.processors}")
        logger.info(f"Initialized KnowledgeBaseLoader, default processor_configs: {self.processor_configs}")

    def register_processor(self, extension: str, processor: object) -> None:
        """
        注册新的文件处理器。
        Args:
            extension (str): 文件扩展名（例如 '.md'）。
            processor (object): 文件处理器实例。
        """
        self.processors[extension.lower()] = processor
        logger.info(f"Registered processor for extension: {extension}")

    def load(self, path: Path | None = None) -> List[str]:
        """
        加载指定路径的文件或目录中的所有文件到向量数据库。
        Args:
            path (Path | None): 文件或目录路径，默认为 self.directory。
        Returns:
            List[str]: 加载的文档 ID 列表。
        """
        if path is None:
            path = self.directory
        logger.info(f"Loading documents from directory: {path}")

        documents = []
        processed_files = set()  # 记录已经处理过的文件

        if path.is_file():
            # 处理单个文件
            logger.info(f"Processing single file: {path}, Suffix: {path.suffix.lower()}")
            if path.suffix.lower() in self.processors:
                processor = self.processors[path.suffix.lower()]
                docs = processor.process(path)
                documents.extend(docs)
            else:
                logger.warning(f"Ignored file: {path} due to unsupported extension")
        elif path.is_dir():
            # 处理目录及其子目录 TODO 分批次insert to vector store
            logger.info(f"Scanning directory: {path}")
            for file_path in path.rglob("*"):
                if file_path.name in processed_files:
                    logger.info(f"Ignored file: {file_path} due to duplicate name")
                    continue
                if file_path.is_file():
                    processed_files.add(file_path.name)
                    logger.info(f"Found file: {file_path}, Suffix: {file_path.suffix.lower()}")
                    if file_path.suffix.lower() in self.processors:
                        logger.info(f"Processing file: {file_path}")
                        processor = self.processors[file_path.suffix.lower()]
                        docs = processor.process(file_path)
                        documents.extend(docs)
                    else:
                        logger.warning(f"Ignored file: {file_path} due to unsupported extension")
        else:
            logger.warning(f"Path {path} is neither a file nor a directory")

        if not documents:
            logger.warning("No valid documents found to load")
            return []

        #  Chroma 新增文档时，若文档 ID 相同，新文档会覆盖旧文档，从而更新向量存储中的文档内容、嵌入向量和元数据
        logger.info(f"Start to load {len(documents)} documents into vector store")

        all_ids = []

        for i in range(0, len(documents), MAX_BATCH_SIZE):
            batch = documents[i:i + MAX_BATCH_SIZE]
            logger.info(f"Start to load a batch of {len(batch)} documents into vector store,batch no:{i}")
            try:
                ids = self.vector_store.add_documents(batch)
                all_ids.extend(ids)
                logger.info(f"Loaded a batch of documents successfully, ids: {ids}")
            except Exception as e:
                logger.error(f"Error loading a batch of documents to vector store: {str(e)}")
                return all_ids

        logger.info(f"Loaded all documents successfully, total number: {len(all_ids)}")
        return all_ids

