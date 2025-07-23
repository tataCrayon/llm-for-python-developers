from fastapi import HTTPException
from typing import AsyncGenerator, Dict

from app.config.deepseek_llm import deep_seek_chat_model,deep_seek_r1_model
from app.schemas.chat_request import ChatRequest
from app.config.logger import setup_logger
from langchain_core.messages import SystemMessage,HumanMessage

logger = setup_logger(__name__)



class DeepSeekService:
    """
    DeepSeek 服务类
    """
    def __init__(self):
        self.chat_model = deep_seek_chat_model()
        self.r1_model = deep_seek_r1_model()
        logger.info('DeepSeek Model Initialized')

    def get_deep_seek_chat(self):
        return self.chat_model

    def get_deep_seek_r1(self):
        return self.r1_model

    def get_deep_seek_model(self,model_name: str):
        if model_name == 'chat':
            return self.chat_model
        elif model_name == 'r1':
            return self.r1_model
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")

    async def generate(self,query :str) -> str:
        """
        生成模型回复,默认使用deepseek-reason

        :param query:
        :return:
        """
        logger.info(f'DeepSeek Request Generate,query:{query}')
        chat_request = ChatRequest(user_message = query,model_name="chat")
        return await self.process_chat(chat_request)

    async def process_chat(self, chat_request:ChatRequest) -> str:
        logger.info(f"DeepSeek chat request: {chat_request}")
        # 应该session_id和msg_id以及token、memory映射等处理，具体代码见java实现，这里略
        try:
            deep_seek_model = self.get_deep_seek_model(chat_request.model_name)
            human_message = chat_request.user_message
            # 将用户消息封装为 HumanMessage 对象
            messages = [HumanMessage(content=human_message)]
            response = await deep_seek_model.agenerate([messages])
            logger.info(f"DeepSeek chat response: {response}")
            # 提取生成的文本内容
            if response.generations:
                generated_text = response.generations[0][0].text
            else:
                generated_text = ""
            return generated_text
        except Exception as e:
            logger.error(f"DeepSeek process_chat error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def stream_chat(self, chat_request: ChatRequest) -> AsyncGenerator[Dict[str, object], None]:
        logger.info(f"DeepSeek stream chat request: {chat_request}")
        full_text = ""
        try:
            deep_seek_model = self.get_deep_seek_model(chat_request.model_name)
            # 开始帧 省略应该有msg_id、session_id等信息
            yield {"event": "start"}
            async for chunk in deep_seek_model.astream(chat_request.user_message):
                data = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_text += data
                yield {"event": "delta", "content": data, "stream": True}
            # 结束帧，包含完整内容
            yield {"event": "end", "stream": False, "status": "FINISHED", "full_text": full_text}
        except Exception as e:
            logger.error(f"DeepSeek stream chat error: {e}")
            yield {"event": "end", "stream": False, "status": "ERROR", "error": str(e), "full_text": full_text}

