from datetime import date, datetime, timezone, timedelta
from typing import Literal, overload
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override


class MemorySemantic(BaseModel):
    # 拒绝模型额外提交 id、created_by 等字段。forbid等用法参考BaseModel
    model_config = ConfigDict(extra="forbid")
    content: str = Field(
        min_length=1,
        description=(
            "完整的规范化记忆正文，保留原文事实和细节。"
            "有明确依据时，将相对日期转换为具体日期，"
            "将代词替换为明确人物名称；不得补充未知事实。"
        ),
    )
    type: Literal[
        "event", "fact", "preference",
        "person", "plan", "opinion"
    ]
    summary: str = Field(min_length=1)
    tags: list[str]
    entities: list[str]
    event_time: date | None = Field(
        description=(
            "事件日期。可以依据原始消息时间解析昨天、前天等表达；"
            "缺少可靠时间依据时返回 null。"
        )
    )
    location: str | None = Field(
        description="原文有明确地点才填写，否则为 null"
    )


def create_memory(
    ret,
    *,
    memory_id: str | None = None,
    raw_content: str,
    semantic: dict,
    source: dict,
    created_by: dict,
    created_at: datetime | None = None,
    processed_by: dict | None,
    metadata: dict | None = None,
):
    if not raw_content.strip():
        raise ValueError("记忆原文不能为空")

    # 校验模型输出，并把日期转换成 JSON 可保存的字符串
    fields = MemorySemantic.model_validate(
        semantic
    ).model_dump(mode="json")

    now = datetime.now(
        timezone(timedelta(hours=8))
    ).isoformat(timespec="seconds")

    memory = {
        "id": memory_id or f"memory_{uuid4().hex}",
        **fields,
        "raw_content": raw_content,
        "source": dict(source),
        "created_by": dict(created_by),
        "processed_by": (
            dict(processed_by) if processed_by else None
        ),
        "created_at": created_at or now,
        "updated_at": now,
        "metadata": metadata or {}
    }

    return memory
