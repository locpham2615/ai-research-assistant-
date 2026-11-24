from pathlib import Path

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder


async def test_embedder_caches_and_only_computes_misses(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    await store.init_db()

    computed_batches: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        computed_batches.append(texts)
        return [[float(len(t)), 0.5, 1.0] for t in texts]

    embedder = TextEmbedder(sqlite_store=store, embed_fn=fake_embed)

    # Lần 1: cả 2 đều miss -> phải tính 2 đoạn
    vecs_1 = await embedder.embed_texts(["alpha", "beta"])
    assert len(vecs_1) == 2
    assert computed_batches == [["alpha", "beta"]]

    # Lần 2: "alpha" hit cache, chỉ "gamma" miss -> chỉ tính đúng 1 đoạn
    vecs_2 = await embedder.embed_texts(["alpha", "gamma"])
    assert vecs_2[0] == vecs_1[0]
    assert computed_batches == [["alpha", "beta"], ["gamma"]]