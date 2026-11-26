from dataclasses import dataclass
from typing import Any

from fastapi import Request

from ai_research_assistant.agent.arxiv_tool import ArxivDiscoveryTool
from ai_research_assistant.agent.research_agent import ResearchAgent
from ai_research_assistant.config import Settings
from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.harness.llm_router import LLMRouter
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.pipelines.chunkers.section_chunker import SectionChunker
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.pipelines.ingest_pipeline import IngestPipeline
from ai_research_assistant.pipelines.parsers.document_parser import DocumentParser
from ai_research_assistant.rag.rag_engine import CitationGroundedRAG
from ai_research_assistant.retrieval.hybrid_retriever import HybridRetriever
from ai_research_assistant.retrieval.query_rewriter import QueryRewriter
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker


@dataclass
class AppContainer:
    settings: Settings
    sqlite_store: SQLiteStore
    research_agent: Any
    rag_engine: Any
    ingest_pipeline: Any
    vector_store: Any = None


async def build_default_container(settings: Settings) -> AppContainer:
    sqlite_store = SQLiteStore(settings.sqlite_db_path)
    await sqlite_store.init_db()

    vector_store = VectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
    )
    vector_store.ensure_collection(vector_size=384)

    router = LLMRouter(settings=settings)
    guard = SchemaGuard(router)
    embedder = TextEmbedder(
        model_name=settings.embedding_model_name,
        sqlite_store=sqlite_store,
    )

    ingest_pipeline = IngestPipeline(
        sqlite_store=sqlite_store,
        vector_store=vector_store,
        parser=DocumentParser(grobid_url=settings.grobid_url),
        chunker=SectionChunker(),
        embedder=embedder,
        schema_guard=guard,
    )

    arxiv_client = getattr(ingest_pipeline, "arxiv_client", None)
    arxiv_tool = (
        ArxivDiscoveryTool(
            arxiv_client=arxiv_client,
            ingest_pipeline=ingest_pipeline,
            download_dir=settings.arxiv_download_dir,
            max_papers=2,
        )
        if arxiv_client is not None
        else None
    )

    research_agent = ResearchAgent.from_components(
        embedder=embedder,
        vector_store=vector_store,
        sqlite_store=sqlite_store,
        schema_guard=guard,
        reranker_model_name=settings.reranker_model_name,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_min_score=settings.rerank_min_score,
        rrf_k=settings.rrf_k,
        max_hops=settings.agent_max_hops,
        max_sub_queries=settings.agent_max_sub_queries,
        fast_path_min_score=settings.agent_fast_path_min_score,
        fast_path_min_chunks=settings.agent_fast_path_min_chunks,
        merged_top_k=settings.agent_merged_top_k,
        max_context_chars=settings.max_context_chars,
        allow_external_arxiv=settings.agent_allow_external_arxiv,
        arxiv_tool=arxiv_tool,
        chat_history_max_turns=settings.chat_history_max_turns,
        chat_history_max_chars=settings.chat_history_max_chars,
    )

    single_shot_retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        sqlite_store=sqlite_store,
        query_rewriter=QueryRewriter(schema_guard=guard),
        retrieval_top_k=settings.retrieval_top_k,
        rrf_k=settings.rrf_k,
    )
    single_shot_reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        top_k=settings.rerank_top_k,
        min_score=settings.rerank_min_score,
    )
    rag_engine = CitationGroundedRAG(
        retriever=single_shot_retriever,
        reranker=single_shot_reranker,
        schema_guard=guard,
        max_context_chars=settings.max_context_chars,
    )

    return AppContainer(
        settings=settings,
        sqlite_store=sqlite_store,
        vector_store=vector_store,
        research_agent=research_agent,
        rag_engine=rag_engine,
        ingest_pipeline=ingest_pipeline,
    )


def get_container(request: Request) -> AppContainer:
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Application container is not initialized")
    return container