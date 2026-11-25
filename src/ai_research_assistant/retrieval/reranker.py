import asyncio
from collections.abc import Callable

from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore[import-untyped]

from ai_research_assistant.retrieval.schemas import RetrievedChunk

ScoreFn = Callable[[str, list[str]], list[float]]


def _format_candidate_text(chunk: RetrievedChunk) -> str:
    if chunk.section.strip():
        return f"{chunk.section.strip()}: {chunk.text}"
    return chunk.text


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        min_score: float = 0.0,
        score_fn: ScoreFn | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score
        self._score_fn = score_fn
        self._model: TextCrossEncoder | None = None

    def _get_model(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        if self._score_fn is not None:
            return [float(s) for s in self._score_fn(query, documents)]
        model = self._get_model()
        return [float(score) for score in model.rerank(query, documents)]

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        limit = top_k if top_k is not None else self.top_k
        threshold = min_score if min_score is not None else self.min_score

        if not query.strip() or not chunks or limit <= 0:
            return []

        documents = [_format_candidate_text(chunk) for chunk in chunks]
        scores = await asyncio.to_thread(self._score_sync, query, documents)

        scored_chunks = [
            chunk.model_copy(update={"rerank_score": score})
            for chunk, score in zip(chunks, scores, strict=True)
            if score >= threshold
        ]

        scored_chunks.sort(
            key=lambda c: (
                c.rerank_score if c.rerank_score is not None else float("-inf"),
                c.rrf_score,
            ),
            reverse=True,
        )
        return scored_chunks[:limit]