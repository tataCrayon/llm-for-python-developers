from typing import Dict

from fastapi import APIRouter, Depends

from app.config.logger import setup_logger
from app.dependencies import log_time
from app.schemas.chat_request import ChatRequest
from app.services.adaptive__rag_service import ChatWithRAGService

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


@router.post("/profiled")
async def rag_chat(request: ChatRequest, k: int | None = None) -> Dict:
    return await chatService.adaptive_chat(request, k)
