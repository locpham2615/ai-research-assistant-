import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ChatRole = Literal["user", "assistant"]

_WHITESPACE_RE = re.compile(r"\s+")


class ChatTurn(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)

    @field_validator("content", mode="before")
    @classmethod
    def _clean_content(cls, value: object) -> str:
        return str(value).strip() if value is not None else ""


class ChatSession(BaseModel):
    session_id: str
    topic_id: str | None = None
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ChatMessage(BaseModel):
    message_id: str
    session_id: str
    role: ChatRole
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    agent_trace: dict[str, Any] | None = None
    is_grounded: bool | None = None
    abstain_reason: str = ""
    created_at: str


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def format_chat_history(
    history: list[ChatTurn] | None,
    max_turns: int = 4,
    max_chars: int = 1500,
    max_message_chars: int = 400,
) -> str:
    """
    Định dạng cửa sổ trượt lịch sử hội thoại cho QueryPlanner:
    - Chỉ giữ tối đa `max_turns` cặp hỏi-đáp gần nhất (2 * max_turns tin nhắn).
    - Cắt gọn từng tin nhắn dài và khóa tổng ngân sách <= `max_chars`.
    - Ưu tiên giữ các lượt mới nhất nhưng xuất ra theo đúng thứ tự thời gian.
    """
    if not history or max_turns <= 0 or max_chars <= 0:
        return ""

    recent = history[-(max_turns * 2) :]
    selected_reversed: list[str] = []
    used = 0

    for turn in reversed(recent):
        label = "User" if turn.role == "user" else "Assistant"
        prefix = f"{label}: "
        body = _truncate_text(turn.content, max_message_chars)
        if not body:
            continue

        line = f"{prefix}{body}"
        sep = 1 if selected_reversed else 0
        if used + sep + len(line) <= max_chars:
            selected_reversed.append(line)
            used += sep + len(line)
            continue

        # Nếu lượt mới nhất vẫn dài hơn phần ngân sách còn lại, cắt gọn cho vừa khít
        remaining = max_chars - used - sep - len(prefix)
        if not selected_reversed and remaining > 20:
            selected_reversed.append(f"{prefix}{_truncate_text(body, remaining)}")
        break

    if not selected_reversed:
        return ""
    return "\n".join(reversed(selected_reversed))