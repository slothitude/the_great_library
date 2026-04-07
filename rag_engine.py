from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import faiss
import networkx as nx
import numpy as np
from ollama import Client

import config
from logger import log

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EngineError(Exception):
    """Base engine error."""


class EmbedError(EngineError):
    """Embedding generation failed."""


class ParseError(EngineError):
    """File parsing failed."""

# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """FAISS + NetworkX RAG engine with Ollama embeddings."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._init_dirs()
        self.client = Client(host=config.OLLAMA_HOST)
        self.embed_dim = self._detect_dim()
        self.index = self._load_or_create_index()
        self.graph = self._load_or_create_graph()
        self.manifest = self._load_manifest()
        self._check_model_change()
        self.next_id: int = self._compute_next_id()
        log.info(
            "Engine ready  dim=%d  model=%s  chunks=%d",
            self.embed_dim,
            config.EMBED_MODEL,
            self.index.ntotal,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_dirs(self) -> None:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.DB_DIR, exist_ok=True)

    def _detect_dim(self) -> int:
        """Auto-detect embedding dimensionality from the model."""
        try:
            resp = self.client.embed(model=config.EMBED_MODEL, input=["probe"])
            dim = len(resp.embeddings[0])
            log.info("Auto-detected embedding dimension: %d", dim)
            return dim
        except Exception as exc:
            raise EmbedError(f"Cannot connect to Ollama / detect dim: {exc}") from exc

    def _load_or_create_index(self) -> faiss.IndexIDMap:
        path = os.path.join(config.DB_DIR, "faiss.index")
        if os.path.exists(path):
            try:
                idx = faiss.read_index(path)
                if isinstance(idx, faiss.IndexIDMap) and idx.ntotal > 0:
                    log.info("Loaded FAISS index with %d vectors", idx.ntotal)
                    return idx
            except Exception as exc:
                log.warning("Failed to load FAISS index, creating new: %s", exc)
        base = faiss.IndexFlatIP(self.embed_dim)
        return faiss.IndexIDMap(base)

    def _load_or_create_graph(self) -> nx.Graph:
        path = os.path.join(config.DB_DIR, "graph.gml")
        if os.path.exists(path):
            try:
                g = nx.read_gml(path)
                log.info("Loaded graph with %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
                return g
            except Exception as exc:
                log.warning("Failed to load graph, creating new: %s", exc)
        return nx.Graph()

    def _load_manifest(self) -> dict[str, Any]:
        path = os.path.join(config.DB_DIR, "manifest.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as exc:
                log.warning("Failed to load manifest: %s", exc)
        return {"files": {}, "model": config.EMBED_MODEL}

    def _check_model_change(self) -> None:
        saved = self.manifest.get("model")
        if saved and saved != config.EMBED_MODEL:
            log.warning(
                "Model changed from '%s' to '%s' — clearing index",
                saved,
                config.EMBED_MODEL,
            )
            base = faiss.IndexFlatIP(self.embed_dim)
            self.index = faiss.IndexIDMap(base)
            self.graph = nx.Graph()
            self.manifest = {"files": {}, "model": config.EMBED_MODEL}

    def _compute_next_id(self) -> int:
        """Figure out the next usable int ID for FAISS (must not collide)."""
        if self.index.ntotal == 0:
            return 1
        # Use range_search or IDArrayContainer to get IDs
        try:
            id_storage = self.index.id_map
            ids = np.zeros(self.index.ntotal, dtype=np.int64)
            for i in range(self.index.ntotal):
                ids[i] = id_storage.at(i)
            return int(ids.max()) + 1
        except Exception:
            return 1

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning unit-normalised vectors."""
        if not texts:
            return np.empty((0, self.embed_dim), dtype=np.float32)
        try:
            resp = self.client.embed(model=config.EMBED_MODEL, input=texts)
            vecs = np.array(resp.embeddings, dtype=np.float32)
        except Exception as exc:
            raise EmbedError(f"Embedding failed: {exc}") from exc
        # normalise to unit length for cosine via inner-product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs /= norms
        return vecs

    def _embed_single(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _parse_file(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext in (".txt", ".md"):
            return Path(path).read_text(encoding="utf-8", errors="replace")
        if ext == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(path)
                pages = [p.extract_text() or "" for p in reader.pages]
                return "\n\n".join(pages)
            except Exception as exc:
                raise ParseError(f"PDF parse error: {exc}") from exc
        if ext == ".csv":
            try:
                with open(path, newline="", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    return "\n".join(" | ".join(row) for row in rows)
            except Exception as exc:
                raise ParseError(f"CSV parse error: {exc}") from exc
        if ext == ".json":
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                return RAGEngine._json_to_text(data)
            except Exception as exc:
                raise ParseError(f"JSON parse error: {exc}") from exc
        raise ParseError(f"Unsupported file type: {ext}")

    @staticmethod
    def _json_to_text(obj: Any, depth: int = 0) -> str:
        parts: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                parts.append(f"{k}: {RAGEngine._json_to_text(v, depth+1)}")
        elif isinstance(obj, list):
            for item in obj:
                parts.append(RAGEngine._json_to_text(item, depth+1))
        else:
            parts.append(str(obj))
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        size = max(1, config.CHUNK_SIZE)
        overlap = min(config.CHUNK_OVERLAP, size - 1)
        chunks: list[str] = []
        i = 0
        while i < len(words):
            chunk = words[i : i + size]
            if chunk:
                chunks.append(" ".join(chunk))
            i += size - overlap
            if i >= len(words):
                break
        return chunks

    @staticmethod
    def _chunk_id(file_path: str, chunk_index: int) -> int:
        """Deterministic int ID from file path + chunk index."""
        h = hashlib.md5(f"{file_path}:{chunk_index}".encode()).hexdigest()[:14]
        return int(h, 16)

    # ------------------------------------------------------------------
    # FAISS helpers
    # ------------------------------------------------------------------

    def _add_to_faiss(self, vecs: np.ndarray, ids: list[int]) -> None:
        with self.lock:
            self.index.add_with_ids(vecs, np.array(ids, dtype=np.int64))

    def _search_faiss(self, vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            scores, ids = self.index.search(vec.reshape(1, -1), k)
        return scores[0], ids[0]

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def _build_edges_for(self, new_ids: list[int]) -> None:
        """Build graph edges for new chunk IDs using FAISS neighbour search."""
        if not new_ids:
            return
        k = config.GRAPH_TOP_K + 1  # +1 because self may be returned
        for cid in new_ids:
            try:
                vec = self._get_vector(cid)
                if vec is None:
                    continue
                scores, nbr_ids = self._search_faiss(vec, k)
                for score, nbr_id in zip(scores, nbr_ids):
                    if nbr_id < 0:
                        continue
                    nbr_id = int(nbr_id)
                    if nbr_id == cid:
                        continue
                    if float(score) >= config.GRAPH_SIM_THRESHOLD:
                        self.graph.add_edge(
                            cid, nbr_id, weight=float(score), source="similarity"
                        )
            except Exception as exc:
                log.warning("Edge build failed for %d: %s", cid, exc)

    def _get_vector(self, chunk_id: int) -> np.ndarray | None:
        """Reconstruct a vector from the FAISS index by ID."""
        try:
            with self.lock:
                # search with the ID itself to reconstruct — use id_map
                idx_pos = self.index.id_map.search(np.array([chunk_id], dtype=np.int64))
                if idx_pos[0][0] < 0:
                    return None
                vec = self.index.index.reconstruct(int(idx_pos[0][0]))
            return vec
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        idx_path = os.path.join(config.DB_DIR, "faiss.index")
        gml_path = os.path.join(config.DB_DIR, "graph.gml")
        man_path = os.path.join(config.DB_DIR, "manifest.json")
        with self.lock:
            faiss.write_index(self.index, idx_path)
        # Convert any non-scalar node/edge attrs for GML compatibility
        clean_graph = nx.Graph()
        for n, ndata in self.graph.nodes(data=True):
            clean_graph.add_node(n, **{k: v for k, v in ndata.items() if isinstance(v, (str, int, float))})
        for u, v, edata in self.graph.edges(data=True):
            clean_graph.add_edge(u, v, **{k: v for k, v in edata.items() if isinstance(v, (str, int, float))})
        nx.write_gml(clean_graph, gml_path)
        self.manifest["model"] = config.EMBED_MODEL
        with open(man_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        log.info("Saved index (%d vectors), graph (%d edges), manifest", self.index.ntotal, self.graph.number_of_edges())

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, force: bool = False) -> dict[str, Any]:
        """Walk data/, ingest new/changed files, build graph edges."""
        stats = {
            "files_scanned": 0,
            "files_new": 0,
            "files_updated": 0,
            "files_skipped": 0,
            "chunks_added": 0,
            "errors": [],
        }
        supported = {".txt", ".md", ".pdf", ".csv", ".json"}
        files_manifest: dict[str, Any] = self.manifest.setdefault("files", {})

        # Walk data dir
        file_list: list[str] = []
        for root, _dirs, filenames in os.walk(config.DATA_DIR):
            for fn in filenames:
                fp = os.path.join(root, fn)
                if Path(fp).suffix.lower() in supported:
                    file_list.append(fp)

        stats["files_scanned"] = len(file_list)
        all_new_ids: list[int] = []

        for fp in file_list:
            try:
                sha = self._file_sha256(fp)
                prev = files_manifest.get(fp)
                if not force and prev and prev.get("sha256") == sha:
                    stats["files_skipped"] += 1
                    continue

                text = self._parse_file(fp)
                chunks = self._chunk_text(text)
                if not chunks:
                    continue

                # Generate chunk IDs
                chunk_ids = [self._chunk_id(fp, i) for i in range(len(chunks))]

                # Batch embed
                all_vecs: list[np.ndarray] = []
                for start in range(0, len(chunks), config.EMBED_BATCH_SIZE):
                    batch = chunks[start : start + config.EMBED_BATCH_SIZE]
                    vecs = self._embed_batch(batch)
                    all_vecs.append(vecs)

                vecs = np.vstack(all_vecs)

                # Store chunk metadata in graph nodes
                for cid, chunk_text in zip(chunk_ids, chunks):
                    self.graph.add_node(
                        cid,
                        file_path=fp,
                        chunk_text=chunk_text,
                        chunk_index=chunks.index(chunk_text),
                    )

                # Add to FAISS
                self._add_to_faiss(vecs, chunk_ids)
                all_new_ids.extend(chunk_ids)

                # Update manifest
                is_new = prev is None
                files_manifest[fp] = {"sha256": sha, "chunks": len(chunks)}
                if is_new:
                    stats["files_new"] += 1
                else:
                    stats["files_updated"] += 1
                stats["chunks_added"] += len(chunks)

            except Exception as exc:
                log.error("Ingest error for %s: %s", fp, exc)
                stats["errors"].append({"file": fp, "error": str(exc)})

        # Build graph edges for all new vectors
        if all_new_ids:
            self._build_edges_for(all_new_ids)

        self.save()
        self.next_id = self._compute_next_id()
        return stats

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        expand_graph: bool = True,
        graph_hops: int | None = None,
    ) -> dict[str, Any]:
        top_k = top_k or config.QUERY_TOP_K
        graph_hops = graph_hops or config.QUERY_HOPS

        qvec = self._embed_single(query_text)
        scores, ids = self._search_faiss(qvec, top_k)

        seed_ids: list[int] = [int(i) for i in ids if i >= 0]
        results: list[dict[str, Any]] = []
        seen: set[int] = set()

        # Vector search results
        for score, cid in zip(scores, ids):
            if cid < 0:
                continue
            cid = int(cid)
            seen.add(cid)
            ndata = self.graph.nodes.get(cid, {})
            results.append({
                "chunk_id": cid,
                "file_path": ndata.get("file_path", ""),
                "chunk_text": ndata.get("chunk_text", ""),
                "score": float(score),
                "source": "vector",
            })

        # Graph expansion
        graph_expanded = False
        if expand_graph and seed_ids and self.graph.number_of_nodes() > 0:
            graph_expanded = True
            visited: set[int] = set(seed_ids)
            frontier: set[int] = set(seed_ids)
            for _ in range(graph_hops):
                next_frontier: set[int] = set()
                for node in frontier:
                    for nbr in self.graph.neighbors(node):
                        if nbr not in visited:
                            visited.add(nbr)
                            next_frontier.add(nbr)
                frontier = next_frontier
                if not frontier:
                    break

            # Score graph-expanded nodes by cosine to query
            expanded = visited - seen
            if expanded:
                exp_vecs: list[tuple[int, float]] = []
                for nid in expanded:
                    vec = self._get_vector(nid)
                    if vec is not None:
                        sim = float(np.dot(qvec, vec))
                        exp_vecs.append((nid, sim))
                exp_vecs.sort(key=lambda x: x[1], reverse=True)
                for nid, sim in exp_vecs[:top_k]:
                    ndata = self.graph.nodes.get(nid, {})
                    results.append({
                        "chunk_id": nid,
                        "file_path": ndata.get("file_path", ""),
                        "chunk_text": ndata.get("chunk_text", ""),
                        "score": sim,
                        "source": "graph",
                    })

        results.sort(key=lambda r: r["score"], reverse=True)
        return {"results": results[: top_k * 2], "graph_expanded": graph_expanded}

    # ------------------------------------------------------------------
    # Stats / Health
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "total_documents": len(self.manifest.get("files", {})),
            "total_chunks": int(self.index.ntotal),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "embed_model": config.EMBED_MODEL,
            "embed_dim": self.embed_dim,
        }

    def health(self) -> dict[str, Any]:
        try:
            self.client.ps()
            return {"status": "ok", "ollama_connected": True}
        except Exception:
            return {"status": "degraded", "ollama_connected": False}
