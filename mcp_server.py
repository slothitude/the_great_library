"""MCP tool server for The Great Library — exposes RAG engine as agent tools."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from logger import log
from rag_engine import RAGEngine, EngineError


@asynccontextmanager
async def server_lifespan(app: FastMCP):
    """Create a single RAGEngine instance, save on shutdown."""
    try:
        engine = RAGEngine()
    except EngineError as exc:
        log.error("MCP server engine init failed: %s", exc)
        raise SystemExit(1) from exc
    log.info("MCP server engine ready  chunks=%d", engine.index.ntotal)
    yield {"engine": engine}
    engine.save()
    log.info("MCP server shut down")


mcp = FastMCP("the-great-library", lifespan=server_lifespan)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def ingest(force: bool = False) -> dict[str, Any]:
    """Ingest documents from the data/ directory into the RAG engine.

    Walks data/, hashes each file, ingests new/changed files, skips unchanged.
    Set force=True to re-ingest all files regardless of changes.

    Returns ingestion statistics: files scanned, new, updated, skipped, errors.
    """
    engine: RAGEngine = mcp.get_context().request_context.lifespan_context["engine"]
    try:
        return await asyncio.to_thread(engine.ingest, force)
    except EngineError as exc:
        return {"error": str(exc)}


@mcp.tool()
async def query(
    query_text: str,
    top_k: int | None = None,
    expand_graph: bool = True,
    graph_hops: int | None = None,
) -> dict[str, Any]:
    """Query the RAG engine for relevant document chunks.

    Embeds the query text, searches the FAISS vector index, and optionally
    expands results via graph traversal for related chunks.

    Args:
        query_text: The search query string.
        top_k: Number of results to return (default from config).
        expand_graph: Whether to expand via graph traversal (default True).
        graph_hops: Number of graph hops for expansion (default from config).

    Returns matching chunks with scores, file paths, and source type.
    """
    if not query_text.strip():
        return {"error": "query_text must not be empty"}
    engine: RAGEngine = mcp.get_context().request_context.lifespan_context["engine"]
    try:
        return await asyncio.to_thread(
            engine.query, query_text, top_k, expand_graph, graph_hops
        )
    except EngineError as exc:
        return {"error": str(exc)}


@mcp.tool()
async def stats() -> dict[str, Any]:
    """Get RAG engine statistics.

    Returns total documents, chunks, graph nodes/edges, embedding model, and dimension.
    """
    engine: RAGEngine = mcp.get_context().request_context.lifespan_context["engine"]
    return engine.stats()


@mcp.tool()
async def health() -> dict[str, Any]:
    """Check RAG engine health status.

    Returns status (ok/degraded) and whether Ollama is connected.
    """
    engine: RAGEngine = mcp.get_context().request_context.lifespan_context["engine"]
    return engine.health()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
