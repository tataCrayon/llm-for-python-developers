from typing import Annotated
from uuid import UUID
from datetime import datetime, time, timedelta
from pydantic import BaseModel
from fastapi import Body

from ..dependencies import log_time

from fastapi import APIRouter,Depends

router = APIRouter(
    prefix="/items",
    tags=["items"],
    dependencies=[Depends(log_time)],
    responses={404: {"description": "Not found"}},
)

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


class User(BaseModel):
    username: str
    full_name: str | None = None

@router.get("/", tags = ["items"])
async def hello():
    return "hello items"


@router.put("/update/{item_id}", tags = ["items"])
async def update_item(
    *,
    item_id: int,
    item: Item,
    user: User,
    importance: Annotated[int, Body(gt=0)],
    q: str | None = None,
):
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    if q:
        results.update({"q": q})
    return results



@router.put("/read/{item_id}", tags = ["items"])
async def read_items(
    item_id: UUID,
    start_datetime: Annotated[datetime, Body()],
    end_datetime: Annotated[datetime, Body()],
    process_after: Annotated[timedelta, Body()],
    repeat_at: Annotated[time | None, Body()] = None,
):
    start_process = start_datetime + process_after
    duration = end_datetime - start_process
    return {
        "item_id": item_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "process_after": process_after,
        "repeat_at": repeat_at,
        "start_process": start_process,
        "duration": duration,
    }