# Angruvadal MCP RAM Server

RAM-backed semantic context store. FastAPI server, runs on CPU, 192GB DDR5 as the memory.

## Quick Start

```bash
pip install fastapi uvicorn sentence-transformers numpy
python3 server.py
```

Server starts on port 8765.

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/store` | POST | Store a text chunk with key |
| `/retrieve` | POST | Semantic search, returns top_k chunks |
| `/list` | GET | List all stored keys |
| `/stats` | GET | Access counts, top accessed |
| `/tools` | GET | OpenAI tool definitions for LLM function calling |
| `/health` | GET | Health check |

## Tool Calling

The `/tools` endpoint returns OpenAI-compatible function definitions. Pass them to any llama.cpp-served model:

```python
tools = requests.get("http://localhost:8765/tools").json()
# Pass to llama.cpp OpenAI-compatible completions endpoint
```

GPT-OSS 20B natively calls `context_retrieve` with correct query formulation.

## Benchmarks (RX 9070 + 192GB DDR5)

| Scale | p50 retrieve | p95 retrieve |
|---|---|---|
| 10 chunks | ~7ms | ~12ms |
| 1,000 chunks | TBD | TBD |
| 5,000 chunks | TBD | TBD |

*Full results coming after stress test.*
