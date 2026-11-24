import asyncio
import hashlib
from collections.abc import Callable

from fastembed import TextEmbedding  # type: ignore[import-untyped]

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.pipelines.schemas import DocumentChunk

EmbedFn = Callable[[list[str]], list[list[float]]]


class TextEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        sqlite_store: SQLiteStore | None = None,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self.model_name = model_name
        self.sqlite_store = sqlite_store
        self._embed_fn = embed_fn
        self._model: TextEmbedding | None = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _compute_sync(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        model = self._get_model()
        return [vec.tolist() for vec in model.embed(texts)]

    async def embed_texts(
        self,
        texts: list[str],
        hashes: list[str] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        text_hashes = hashes or [
            hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts
        ]
        vectors: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        # 1. Tra cứu cache trong SQLite trước
        for idx, text_hash in enumerate(text_hashes):
            cached = (
                await self.sqlite_store.get_cached_embedding(text_hash)
                if self.sqlite_store is not None
                else None
            )
            if cached is not None:
                vectors[idx] = cached
            else:
                miss_indices.append(idx)
                miss_texts.append(texts[idx])

        # 2. Chỉ tính embedding cho các đoạn cache miss
        if miss_texts:
            computed = await asyncio.to_thread(self._compute_sync, miss_texts)
            for idx, vec in zip(miss_indices, computed, strict=True):
                vectors[idx] = vec
                if self.sqlite_store is not None:
                    await self.sqlite_store.set_cached_embedding(text_hashes[idx], vec)

        return [v for v in vectors if v is not None]

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        texts = [c.text for c in chunks]
        hashes = [c.text_hash for c in chunks]
        return await self.embed_texts(texts, hashes=hashes)

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed_texts([query])
        return vectors[0]