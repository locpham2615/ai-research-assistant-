from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)


class VectorStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "research_papers",
        in_memory: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:") if in_memory else QdrantClient(url=url)

    def ensure_collection(self, vector_size: int = 384) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(self, points: list[PointStruct]) -> None:
        if not points:
            return
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Dùng cho cơ chế Delete-then-Upsert khi người dùng Update tài liệu."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if doc_ids is not None and len(doc_ids) == 0:
            return []

        conditions: list[Condition] = []
        if topic_id is not None:
            conditions.append(
                FieldCondition(key="topic_id", match=MatchValue(value=topic_id))
            )
        if doc_ids is not None:
            conditions.append(
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            )

        query_filter = Filter(must=conditions) if conditions else None
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return [
            {
                "id": str(point.id),
                "score": float(point.score),
                "payload": point.payload or {},
            }
            for point in response.points
        ]