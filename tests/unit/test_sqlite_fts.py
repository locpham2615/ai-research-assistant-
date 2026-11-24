from pathlib import Path

from ai_research_assistant.db.sqlite import SQLiteStore, build_fts5_query


def test_build_fts5_query_sanitizes_special_chars() -> None:
    q = build_fts5_query("Scaled dot-product (d_k) NOT BLEU-4? /")
    assert q == '"scaled" OR "dot" OR "product" OR "d_k" OR "not" OR "bleu" OR "4"'
    assert build_fts5_query("   --- ???  ") == ""


async def test_sqlite_fts5_bm25_search_filter_and_delete(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    await store.init_db()

    await store.upsert_chunks(
        [
            {
                "chunk_id": "c1",
                "doc_id": "doc_attn",
                "topic_id": "transformers",
                "title": "Attention Is All You Need",
                "section": "Scaled Dot-Product Attention",
                "chunk_index": 0,
                "text": "We scale the dot products by 1 over sqrt(d_k) before softmax.",
                "text_hash": "h1",
            },
            {
                "chunk_id": "c2",
                "doc_id": "doc_attn",
                "topic_id": "transformers",
                "title": "Attention Is All You Need",
                "section": "Training",
                "chunk_index": 1,
                "text": "We used the Adam optimizer with warmup steps.",
                "text_hash": "h2",
            },
            {
                "chunk_id": "c3",
                "doc_id": "doc_other",
                "topic_id": "vision",
                "title": "ViT Paper",
                "section": "Attention",
                "chunk_index": 0,
                "text": "Dot-product attention applied to image patches.",
                "text_hash": "h3",
            },
        ]
    )

    # Tìm với ký tự đặc biệt (dấu gạch nối, ngoặc) + lọc đúng topic_id
    hits = await store.search_bm25(
        "scaled dot-product (d_k) softmax",
        topic_id="transformers",
        limit=5,
    )
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["section"] == "Scaled Dot-Product Attention"
    assert isinstance(hits[0]["lexical_score"], float)

    # Lọc theo doc_ids
    filtered = await store.search_bm25(
        "attention",
        doc_ids=["doc_other"],
        limit=5,
    )
    assert [h["chunk_id"] for h in filtered] == ["c3"]

    # Xóa theo doc_id -> trigger phải gỡ sạch khỏi chunks_fts
    await store.delete_chunks_by_doc_id("doc_attn")
    after_delete = await store.search_bm25("softmax", topic_id="transformers", limit=5)
    assert after_delete == []