# The Great Library

Local RAG engine with FAISS vector search, NetworkX graph traversal, and Ollama embeddings — exposed as a FastAPI HTTP API.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place documents in data/
cp ~/my-docs/*.pdf data/

# Start the server
python main.py
```

Requires [Ollama](https://ollama.com) running with the `nomic-embed-text` model pulled:

```bash
ollama pull nomic-embed-text
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://100.84.161.63:11434` | Ollama instance URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBED_BATCH_SIZE` | `32` | Batch size for embed calls |
| `CHUNK_SIZE` | `512` | Chunking window (words) |
| `CHUNK_OVERLAP` | `64` | Chunk overlap (words) |
| `GRAPH_TOP_K` | `5` | Neighbors per node for graph edges |
| `GRAPH_SIM_THRESHOLD` | `0.7` | Min cosine similarity for graph edge |
| `QUERY_TOP_K` | `5` | Default number of search results |
| `QUERY_HOPS` | `2` | Default graph expansion hops |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `DATA_DIR` | `./data` | Document source directory |
| `DB_DIR` | `./db` | Persisted index/graph/manifest |

## API Endpoints

### `POST /ingest`

Walk `data/`, hash each file, ingest new/changed files, skip unchanged.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

### `POST /query`

Embed a query, search FAISS, optionally expand via graph traversal.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the Library of Alexandria?", "top_k": 5, "expand_graph": true}'
```

### `GET /stats`

```bash
curl http://localhost:8000/stats
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

## MCP Server

The Great Library also exposes its RAG capabilities as MCP tools via stdio transport, so any LLM agent (Claude Code, Cursor, Copilot, etc.) can ingest and query documents as a tool call.

### Install

```bash
pip install -r requirements.txt   # includes mcp>=1.0
```

### Run

```bash
python mcp_server.py
```

Starts an MCP stdio server. It logs engine init to stderr and waits for JSON-RPC on stdin.

### Tools

| Tool | Description |
|------|-------------|
| `ingest(force=False)` | Ingest documents from `data/` into the RAG engine |
| `query(query_text, top_k, expand_graph, graph_hops)` | Search for relevant document chunks |
| `stats()` | Engine statistics (documents, chunks, graph size) |
| `health()` | Health check (status + Ollama connectivity) |

### Claude Code Configuration

Add to your `claude_desktop_config.json` or `.claude/settings.json`:

```json
{
  "mcpServers": {
    "the-great-library": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/the_great_library"
    }
  }
}
```

### Other MCP Clients

Any MCP-compatible client can launch the server with `python mcp_server.py` as the command. The server speaks JSON-RPC over stdio per the [MCP specification](https://modelcontextprotocol.io).

## Supported File Types

- `.txt` / `.md` — plain text
- `.pdf` — via pypdf
- `.csv` — headers + rows as text
- `.json` — recursive string extraction / JSONL

## Architecture

```
data/          → source documents (PDF, TXT, MD, CSV, JSON)
db/            → faiss.index, graph.gml, manifest.json (auto-created)
config.py      → env-based configuration
logger.py      → Rich logger
rag_engine.py  → core engine (FAISS, NetworkX, Ollama, manifest)
main.py        → FastAPI app with async endpoints
```

- **FAISS `IndexIDMap(IndexFlatIP)`** — inner product on unit vectors = cosine similarity
- **NetworkX graph** — edges built via FAISS neighbor search (O(n×K), not O(n²))
- **SHA-256 manifest** — incremental ingestion skips unchanged files
- **Thread-safe** — single lock wraps all FAISS operations, CPU work offloaded via `asyncio.to_thread()`
- **Auto-detect dimension** — embedding size discovered from model at startup
