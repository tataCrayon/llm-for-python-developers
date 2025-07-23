from pydantic import BaseModel
from fastapi import APIRouter,Depends
from app.dependencies import log_time
from app.utils.str_util import StrUtil

from app.config.logger import setup_logger


logger = setup_logger(__name__)


router = APIRouter(
    dependencies=[Depends(log_time)]
)

class TextInfo(BaseModel):
    char_size: int
    word_size: int
    line_size: int

@router.post("/analyze-text")
async def analyze_text(text: str):
    logger.info(f"开始分析文本: {text}")
    text_info_dict = StrUtil.text_analyzer(text)
    logger.info(f"文本分析结果: {text_info_dict}")
    return TextInfo(**text_info_dict)
