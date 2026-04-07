# CLAUDE.md — The Great Library

## Project Overview
Local RAG engine: FAISS vector search + NetworkX graph traversal + Ollama embeddings, exposed as a FastAPI HTTP API and an MCP tool server.

## How to Run
```bash
pip install -r requirements.txt
python main.py          # FastAPI on :8000
python mcp_server.py    # MCP stdio server (for agent integration)
```

Requires Ollama running with `nomic-embed-text` pulled.

## How to Test
```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"force": false}'
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "test", "top_k": 5}'
curl http://localhost:8000/stats
curl http://localhost:8000/health
```

## Key Files
| File | Purpose |
|------|---------|
| `config.py` | Env-based configuration (`.env` file) |
| `logger.py` | Rich console logger |
| `rag_engine.py` | Core engine: `RAGEngine` class, `EngineError` base exception |
| `main.py` | FastAPI app with `/ingest`, `/query`, `/stats`, `/health` |
| `mcp_server.py` | MCP stdio server wrapping `RAGEngine` for agent use |
| `data/` | Source documents (PDF, TXT, MD, CSV, JSON) |
| `db/` | Persisted index, graph, manifest (auto-created) |

## Architecture
- **FAISS `IndexIDMap(IndexFlatIP)`** — inner product on unit vectors = cosine similarity
- **NetworkX graph** — edges via FAISS neighbor search, not O(n²)
- **SHA-256 manifest** — incremental ingestion skips unchanged files
- **Thread-safe** — single lock wraps all FAISS operations; CPU work offloaded via `asyncio.to_thread()`
- **Embedding dimension auto-detected** from model at startup

## Patterns
- All engine errors raise `EngineError` (or subclass `EmbedError`/`ParseError`)
- API endpoints catch `EngineError` and return HTTP 500
- MCP tools catch `EngineError` and return `{"error": "..."}` dicts
- Engine methods are synchronous; async wrappers use `asyncio.to_thread()`

## Gotchas
- Changing `EMBED_MODEL` clears the entire index (detected at startup)
- Chunk IDs are deterministic MD5-derived ints (not sequential)
- Graph edge attributes must be scalar types for GML serialization
- `db/` directory is gitignored — it's runtime state
- Ollama must be reachable at `OLLAMA_HOST` before starting
