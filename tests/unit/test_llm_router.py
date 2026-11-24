import asyncio
from types import SimpleNamespace
from typing import Any

from ai_research_assistant.config import Settings
from ai_research_assistant.harness.llm_router import LLMRouter, TaskType


class FakeCompletions:
    def __init__(self, delay: float = 0.05, reply: str = "ok") -> None:
        self.delay = delay
        self.reply = reply
        self.active = 0
        self.max_active = 0

    async def create(self, **kwargs: Any) -> Any:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def _make_fake_client(completions: FakeCompletions) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


async def test_local_semaphore_serializes_gpu_calls() -> None:
    local_completions = FakeCompletions(delay=0.05, reply="local-ok")
    router = LLMRouter(
        settings=Settings(cloud_llm_api_key="test-key"),
        local_client=_make_fake_client(local_completions),
        cloud_client=_make_fake_client(FakeCompletions(reply="cloud-ok")),
        local_concurrency=1,
    )

    # Bắn 3 request local cùng lúc
    results = await asyncio.gather(
        router.complete(TaskType.SCREENING, [{"role": "user", "content": "a"}]),
        router.complete(TaskType.QUERY_REWRITE, [{"role": "user", "content": "b"}]),
        router.complete(TaskType.EXTRACT_CARD, [{"role": "user", "content": "c"}]),
    )

    assert all(r.provider == "local" for r in results)
    # Tại mọi thời điểm chỉ có tối đa 1 request chạy trên GPU
    assert local_completions.max_active == 1


async def test_cloud_task_routes_to_cloud() -> None:
    router = LLMRouter(
        settings=Settings(cloud_llm_api_key="test-key"),
        local_client=_make_fake_client(FakeCompletions(reply="local-ok")),
        cloud_client=_make_fake_client(FakeCompletions(reply="cloud-ok")),
    )

    res = await router.complete(TaskType.RAG_SYNTHESIS, [{"role": "user", "content": "hi"}])
    assert res.provider == "cloud"
    assert res.content == "cloud-ok"