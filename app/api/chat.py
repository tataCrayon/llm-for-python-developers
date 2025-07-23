from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.dependencies import log_time

from app.schemas.chat_request import ChatRequest
from app.schemas.chat_response import ChatResponse

from app.services.chat_service import DeepSeekService
import json
from app.config.logger import setup_logger


"""
具体对话设计在llm-for-java-developers已经有了，这里简单写两个一次性对话。
这个项目重点还是在llm-for-java-developers没有写的RAG上
"""


logger = setup_logger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(log_time)],
    responses={404: {"description": "Not found"}},
)

deepseek_service = DeepSeekService()

@router.post("/ds/chat")
async def ds_chat(request: ChatRequest):
    if getattr(request, 'stream', False):
        async def event_generator():
            async for data in deepseek_service.stream_chat(request):
                # 处理数据，这里简单返回
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        response = await deepseek_service.process_chat(request)
        return ChatResponse(content=response)

