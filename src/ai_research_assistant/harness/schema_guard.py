import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ai_research_assistant.harness.llm_router import LLMRouter, TaskType

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class SchemaValidationFailedError(Exception):
    """Raised when LLM output fails Pydantic validation after all repair attempts."""


def _extract_json_str(raw_text: str) -> str:
    text = raw_text.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


class SchemaGuard:
    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    async def complete_structured(
        self,
        task_type: TaskType,
        messages: list[dict[str, Any]],
        response_model: type[T],
        max_retries: int = 2,
        temperature: float = 0.0,
        allow_fallback: bool = False,
    ) -> T:
        schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        working_messages = list(messages)
        working_messages.append(
            {
                "role": "system",
                "content": (
                    "Return ONLY a valid JSON object matching this exact JSON schema. "
                    "Do not include markdown fences or extra commentary.\n"
                    f"Schema: {schema_json}"
                ),
            }
        )

        last_error = ""
        for attempt in range(max_retries + 1):
            response = await self.router.complete(
                task_type=task_type,
                messages=working_messages,
                temperature=temperature,
                allow_fallback=allow_fallback,
            )
            cleaned = _extract_json_str(response.content)
            try:
                return response_model.model_validate_json(cleaned)
            except ValidationError as exc:
                last_error = str(exc)
                if attempt == max_retries:
                    break
                working_messages.append({"role": "assistant", "content": response.content})
                working_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous JSON was invalid:\n{last_error}\n"
                            "Fix the errors and return ONLY the corrected valid JSON object."
                        ),
                    }
                )

        raise SchemaValidationFailedError(
            f"Failed to parse {response_model.__name__} after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )