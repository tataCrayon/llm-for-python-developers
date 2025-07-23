from fastapi import FastAPI

from app.config.rag_prompt import RAGPromptConfig
from app.api import items,analyze_text, chat_router,rag_agent_router

from app.config.logger import setup_logger
logger = setup_logger('init_test')
logger.info("=== 启动测试日志 ===")  # 检查是否显示

# app = FastAPI(dependencies=[Depends(get_query_token)])
app = FastAPI()
app.include_router(items.router, tags=["items"])
app.include_router(analyze_text.router, tags=["text"])
app.include_router(chat_router, tags=["chat"])
app.include_router(rag_agent_router, tags=["rag-agent"])

# ========= 初始化 =========

## prompt初始化
rag_prompt_path = "F:/Projects/PythonProjects/llm-for-python-developers/data/prompts/rag_prompt_zh.txt"
RAGPromptConfig.set_rag_prompt_path(rag_prompt_path)



@app.get("/")
async def root():
    return {"message": "Hello, World!"}
