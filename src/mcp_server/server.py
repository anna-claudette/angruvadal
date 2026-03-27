#!/usr/bin/env python3
"""
Angruvadal MCP RAM Server
RAM-backed context store with semantic retrieval for local LLM inference.
Exposes OpenAI-compatible tool definitions that llama.cpp can call.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, json, time, numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional
import threading

app = FastAPI(title="Angruvadal MCP Server")

# ── In-RAM context store ──────────────────────────────────────────
# Key → {text, embedding, timestamp, access_count}
context_store = {}
store_lock = threading.Lock()

# Small fast embedding model (runs on CPU, ~100MB)
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, 14k tok/s on CPU
print("Embedding model ready.")

# ── Pydantic models ───────────────────────────────────────────────
class StoreRequest(BaseModel):
    key: str
    text: str
    metadata: Optional[dict] = {}

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3
    min_score: float = 0.0

# ── Core operations ───────────────────────────────────────────────
@app.post("/store")
def store(req: StoreRequest):
    embedding = embedder.encode(req.text, convert_to_numpy=True)
    with store_lock:
        context_store[req.key] = {
            "text": req.text,
            "embedding": embedding,
            "metadata": req.metadata,
            "timestamp": time.time(),
            "access_count": 0,
        }
    return {"status": "stored", "key": req.key, "chars": len(req.text)}

@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    if not context_store:
        return {"results": [], "total_stored": 0}
    
    query_emb = embedder.encode(req.query, convert_to_numpy=True)
    
    scores = []
    with store_lock:
        for key, item in context_store.items():
            sim = float(np.dot(query_emb, item["embedding"]) /
                       (np.linalg.norm(query_emb) * np.linalg.norm(item["embedding"]) + 1e-9))
            scores.append((key, sim, item["text"]))
            item["access_count"] += 1
    
    scores.sort(key=lambda x: x[1], reverse=True)
    results = [
        {"key": k, "score": round(s, 4), "text": t}
        for k, s, t in scores[:req.top_k]
        if s >= req.min_score
    ]
    
    return {"results": results, "total_stored": len(context_store)}

@app.get("/list")
def list_keys():
    with store_lock:
        return {
            "keys": list(context_store.keys()),
            "count": len(context_store),
            "total_chars": sum(len(v["text"]) for v in context_store.values()),
            "ram_bytes_approx": sum(
                len(v["text"]) + v["embedding"].nbytes 
                for v in context_store.values()
            )
        }

@app.get("/stats")
def stats():
    with store_lock:
        if not context_store:
            return {"count": 0}
        top_accessed = sorted(
            context_store.items(), 
            key=lambda x: x[1]["access_count"], 
            reverse=True
        )[:5]
        return {
            "count": len(context_store),
            "top_accessed": [{"key": k, "count": v["access_count"]} for k, v in top_accessed],
        }

@app.get("/health")
def health():
    return {"status": "ok", "store_count": len(context_store)}

# ── OpenAI tool definitions (for llama.cpp function calling) ──────
@app.get("/tools")
def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "context_store",
                "description": "Store a piece of text in the RAM memory server for later retrieval. Use this to remember important information across a long conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Unique identifier for this memory chunk"},
                        "text": {"type": "string", "description": "The text content to store in RAM memory"},
                    },
                    "required": ["key", "text"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "context_retrieve",
                "description": "Search the RAM memory server for relevant context. Returns the most semantically similar stored chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What you're looking for in memory"},
                        "top_k": {"type": "integer", "description": "Number of results to return (default 3)", "default": 3}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

if __name__ == "__main__":
    print("Angruvadal MCP RAM Server starting on port 8765...")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
