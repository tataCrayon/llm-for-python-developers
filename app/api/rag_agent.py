from typing import Dict

from fastapi import APIRouter, Depends

from app.db.vector_store.chroma_store import ChromaVectorStore
from app.dependencies import log_time

from app.schemas.chat_request import ChatRequest

from app.config.logger import setup_logger
from app.services.chat_service import DeepSeekService
from app.services.chat_with_rag_service import ChatWithRAGService
from app.services.knowledge_level_analysis_service import KnowledgeLevelAnalysisService

"""
RAG对外接口
"""

logger = setup_logger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=["rag-agent"],
    dependencies=[Depends(log_time)],
    responses={404: {"description": "Not found"}},
)

chatService = ChatWithRAGService()

@router.post("/chat")
async def rag_chat(request: ChatRequest):
    return await chatService.rag_pipeline(request.user_message)


