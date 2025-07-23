from typing import Annotated
from datetime import datetime
from fastapi import Header, HTTPException
from app.config.logger import setup_logger


logger = setup_logger(__name__)

# 依赖注入定义

async def log_time():
    logger.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def get_token_header(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def get_query_token(token: str):
    if token != "jessica":
        raise HTTPException(status_code=400, detail="No Jessica token provided")