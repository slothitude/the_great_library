import os
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


OLLAMA_HOST = _env("OLLAMA_HOST", "http://100.84.161.63:11434")
# Ensure the host has a scheme prefix (system OLLAMA_HOST may omit http://)
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"
# 0.0.0.0 is a listen address, not reachable for connect — use localhost
if "://0.0.0.0" in OLLAMA_HOST:
    OLLAMA_HOST = OLLAMA_HOST.replace("0.0.0.0", "127.0.0.1")
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE = _env_int("EMBED_BATCH_SIZE", 32)
CHUNK_SIZE = _env_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 64)
GRAPH_TOP_K = _env_int("GRAPH_TOP_K", 5)
GRAPH_SIM_THRESHOLD = _env_float("GRAPH_SIM_THRESHOLD", 0.7)
QUERY_TOP_K = _env_int("QUERY_TOP_K", 5)
QUERY_HOPS = _env_int("QUERY_HOPS", 2)
API_HOST = _env("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)
DATA_DIR = _env("DATA_DIR", "./data")
DB_DIR = _env("DB_DIR", "./db")
