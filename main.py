from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from logger import log
from rag_engine import RAGEngine, EngineError


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    force: bool = False


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    expand_graph: bool = True
    graph_hops: int | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine = RAGEngine()
    except EngineError as exc:
        log.error("Engine init failed: %s", exc)
        raise SystemExit(1) from exc
    app.state.engine = engine
    log.info("Great Library API starting on %s:%d", config.API_HOST, config.API_PORT)
    yield
    engine.save()
    log.info("Great Library API shut down")


app = FastAPI(title="The Great Library", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest(req: IngestRequest):
    engine: RAGEngine = app.state.engine
    try:
        result = await asyncio.to_thread(engine.ingest, req.force)
        return result
    except EngineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    engine: RAGEngine = app.state.engine
    try:
        result = await asyncio.to_thread(
            engine.query,
            req.query,
            req.top_k,
            req.expand_graph,
            req.graph_hops,
        )
        return result
    except EngineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
async def stats():
    return app.state.engine.stats()


@app.get("/health")
async def health():
    return app.state.engine.health()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
    )
