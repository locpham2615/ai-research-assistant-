import asyncio
import logging
import time
from enum import StrEnum
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ai_research_assistant.config import Settings, get_settings

logger = logging.getLogger(__name__)


class TaskType(StrEnum):
    # Local-first tasks (short, repetitive, $0 cost)
    SCREENING = "screening"
    QUERY_REWRITE = "query_rewrite"
    EXTRACT_CARD = "extract_card"
    COMPACTION = "compaction"

    # Cloud-first tasks (long context, multi-doc synthesis)
    RAG_SYNTHESIS = "rag_synthesis"
    MATRIX_REDUCE = "matrix_reduce"
    DEEP_ANALYSIS = "deep_analysis"

    RAG_ANSWER = "rag_answer"


LOCAL_TASKS: frozenset[TaskType] = frozenset(
    {
        TaskType.SCREENING,
        TaskType.QUERY_REWRITE,
        TaskType.RAG_ANSWER,
        TaskType.EXTRACT_CARD,
        TaskType.COMPACTION,
    }
)


class LLMResponse(BaseModel):
    content: str
    provider: Literal["local", "cloud"]
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        local_client: AsyncOpenAI | None = None,
        cloud_client: AsyncOpenAI | None = None,
        local_concurrency: int = 1,
    ) -> None:
        self.settings = settings or get_settings()
        self.local_client = local_client or AsyncOpenAI(
            base_url=self.settings.local_llm_base_url,
            api_key="not-needed",
        )
        self.cloud_client = cloud_client or AsyncOpenAI(
            base_url=self.settings.cloud_llm_base_url,
            api_key=self.settings.cloud_llm_api_key or "missing-key",
        )
        # Hardware guard: serialize local GPU inference to prevent 4GB VRAM OOM
        self._local_semaphore = asyncio.Semaphore(local_concurrency)

    def _resolve_provider(self, task_type: TaskType) -> Literal["local", "cloud"]:
        return "local" if task_type in LOCAL_TASKS else "cloud"

    async def _call_local(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        async with self._local_semaphore:
            start = time.perf_counter()
            coro = self.local_client.chat.completions.create(
                model=self.settings.local_llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            completion = await asyncio.wait_for(
                coro,
                timeout=self.settings.local_llm_timeout_sec,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            usage = completion.usage
            return LLMResponse(
                content=completion.choices[0].message.content or "",
                provider="local",
                model=self.settings.local_llm_model,
                latency_ms=round(latency_ms, 2),
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _call_cloud(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        start = time.perf_counter()
        completion = await self.cloud_client.chat.completions.create(
            model=self.settings.cloud_llm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.settings.cloud_llm_timeout_sec,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = completion.usage
        return LLMResponse(
            content=completion.choices[0].message.content or "",
            provider="cloud",
            model=self.settings.cloud_llm_model,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    async def complete(
        self,
        task_type: TaskType,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
    ) -> LLMResponse:
        provider = self._resolve_provider(task_type)

        if provider == "local":
            try:
                return await self._call_local(messages, temperature, max_tokens)
            except (APIConnectionError, APITimeoutError, TimeoutError) as exc:
                if allow_fallback and self.settings.cloud_llm_api_key:
                    logger.warning(
                        "Local LLM failed (%s) for task=%s. Falling back to cloud.",
                        type(exc).__name__,
                        task_type,
                    )
                    return await self._call_cloud(messages, temperature, max_tokens)
                raise

        # Cloud tasks never fall back to local 4B model
        return await self._call_cloud(messages, temperature, max_tokens)