from pydantic import BaseModel
from typing import Optional, Literal


class ChatResponse(BaseModel):
    """
    对话请求响应
    参考了DeepSeek、QWen做法
    """
    # 消息ID (参考DS设计，使用自增整数)
    msg_id: int | None = None

    # 父消息ID (用于对话树结构)
    parent_msg_id: int | None = None

    # 会话ID
    session_id: str | None = None

    # 消息内容
    content: str | None = None

    # 是否为流式响应
    stream: bool | None = False

    # 状态 (参考DS设计: WIP/FINISHED)
    # status: Optional[Literal["WIP", "FINISHED"]]  | None = None
    status: str | None = None