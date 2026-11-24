from typing import Any

import pytest
from pydantic import BaseModel

from ai_research_assistant.harness.llm_router import LLMResponse, TaskType
from ai_research_assistant.harness.schema_guard import (
    SchemaGuard,
    SchemaValidationFailedError,
)


class SampleOutput(BaseModel):
    is_relevant: bool
    score: float


class SequenceFakeRouter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, **kwargs: Any) -> LLMResponse:
        content = self._responses[self.calls]
        self.calls += 1
        return LLMResponse(
            content=content,
            provider="local",
            model="fake-local",
            latency_ms=1.0,
        )


async def test_schema_guard_auto_repairs_invalid_json() -> None:
    fake_router = SequenceFakeRouter(
        responses=[
            '{"is_relevant": true}',  # Lần 1: thiếu trường score
            '```json\n{"is_relevant": true, "score": 0.95}\n```',  # Lần 2: hợp lệ
        ]
    )
    guard = SchemaGuard(router=fake_router)  # type: ignore[arg-type]

    result = await guard.complete_structured(
        task_type=TaskType.SCREENING,
        messages=[{"role": "user", "content": "test"}],
        response_model=SampleOutput,
        max_retries=2,
    )

    assert result.is_relevant is True
    assert result.score == 0.95
    assert fake_router.calls == 2


async def test_schema_guard_raises_after_max_retries() -> None:
    fake_router = SequenceFakeRouter(
        responses=["not json", "still bad", "broken again"]
    )
    guard = SchemaGuard(router=fake_router)  # type: ignore[arg-type]

    with pytest.raises(SchemaValidationFailedError):
        await guard.complete_structured(
            task_type=TaskType.SCREENING,
            messages=[{"role": "user", "content": "test"}],
            response_model=SampleOutput,
            max_retries=2,
        )

    assert fake_router.calls == 3