from pydantic import BaseModel

class ChatReqOptions(BaseModel):
    """
    对话参数选型
    参考了Gemini设计，提供了部分
    """
    max_tokens: str | None = None
    temperature: float | None = None
    top_p: float | None = None

class ChatRequest(BaseModel):
    """
    对话请求传输对象 参考了DeepSeek、QWen做法
    """
    session_id: str | None = None
    parent_msg_id: int | None = None
    user_message: str
    stream: bool | None = False
    options: ChatReqOptions | None = None
    model_name: str | None = "chat"



