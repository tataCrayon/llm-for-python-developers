import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.embedding_model import EmbeddingModelComponent
from app.config.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VECTOR_STORE_PATH = Path(os.getenv("CHROMA_PERSIST_DIR", PROJECT_ROOT / "data" / "vector_store"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")


class ChromaVectorStore:
    """
    Chroma 向量存储，支持增删查改和持久化。
    """

    def __init__(
            self,
            persist_directory: Path = DEFAULT_VECTOR_STORE_PATH,
            collection_name: str = CHROMA_COLLECTION_NAME,
            embeddings: HuggingFaceEmbeddings | None = None
    ):
        logger.info("Start to initialize ChromaVectorStore")
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._embeddings = embeddings or EmbeddingModelComponent.get_default()
        self._vector_store = self._load_or_create_vector_store()

    def _load_or_create_vector_store(self) -> Chroma:
        """
        加载|创建向量数据库
        """
        if self._persist_directory.exists():
            logger.info(f"Loading existing Chroma database from {self._persist_directory}")
        else:
            logger.info(f"Creating new Chroma database at {self._persist_directory}")
        return Chroma(
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_directory),
            collection_name=self._collection_name
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量数据库
        """
        ids = []
        for doc in documents:
            if 'id' in doc.metadata and doc.metadata['id']:
                ids.append(str(doc.metadata['id']))
            else:
                logger.error(f"Document missing ID: {doc.metadata}")
                raise ValueError(f"Document missing ID: {doc.metadata}")
        try:
            added_ids = self._vector_store.add_documents(documents, ids=ids)
            logger.info(f"Added {len(documents)} documents with IDs: {added_ids}")
            return added_ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []

    def search_documents(self,
                         query: str,
                         k: int = 5,
                         meta_filter: Optional[dict[str, str]] = None) -> List[Document]:
        """
        查询向量,
        """
        try:
            results = self._vector_store.similarity_search(query, k=k, filter=meta_filter)
            logger.info(f"Retrieved {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def delete_documents(self, ids: List[str]) -> bool:
        try:
            self._vector_store.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents with IDs: {ids}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update_document(self, doc_id: str, new_document: Document) -> bool:
        try:
            self._vector_store.delete(ids=[doc_id])
            new_document.metadata['id'] = doc_id
            self._vector_store.add_documents([new_document])
            logger.info(f"Updated document with ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document with ID {doc_id}: {e}")
            return False



    @property
    def persist_directory(self) -> Path:
        return self._persist_directory

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings

    @property
    def vector_store(self) -> Chroma:
        return self._vector_store
