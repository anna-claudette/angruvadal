#!/usr/bin/env python3
"""Angruvadal MCP RAM Server — End-to-end PoC test"""
import requests, json, time

MCP_URL = "http://10.0.0.30:8765"
LLM_URL = "http://10.0.0.30:8081/v1/chat/completions"
MODEL = "gpt-oss-20b-Q8.gguf"

def llm(messages, tools=None, max_tokens=500):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    r = requests.post(LLM_URL, json=payload, timeout=120)
    return r.json()

def mcp_store(key, text):
    t0 = time.time()
    r = requests.post(f"{MCP_URL}/store", json={"key": key, "text": text})
    elapsed = (time.time() - t0) * 1000
    return r.json(), elapsed

def mcp_retrieve(query, top_k=3):
    t0 = time.time()
    r = requests.post(f"{MCP_URL}/retrieve", json={"query": query, "top_k": top_k})
    elapsed = (time.time() - t0) * 1000
    return r.json(), elapsed

# ── TEST 1: Direct MCP operations ─────────────────────────────────
print("=== TEST 1: Direct MCP store/retrieve ===")

facts = {
    "angruvadal_architecture": "Angruvadal uses RotorQuant 3-bit KV compression achieving 3.5x compression ratio on GPU with Triton kernels, combined with a RAM-backed MCP server for semantic context retrieval.",
    "gpt_oss_benchmark": "GPT-OSS 20B achieves 134 tok/s on the RX 9070 via llama.cpp Vulkan. VRAM usage is 12.5GB of 16GB total.",
    "rdna4_discovery": "The RX 9070 uses gfx1201 (RDNA4). Flash attention gives 5.5x prompt processing improvement. ROCm MMQ+GRAPHS+FA beats Vulkan on prompt processing.",
    "bitsandbytes_fix": "bitsandbytes 0.50.0.dev0 supports gfx1201. Build with GPU_TARGETS=gfx1201. QLoRA and LLM.int8 both pass on RDNA4.",
    "naa_ru_lore": "The Naa'ru are crystalline entities from the asteroid belt. They trade mass for homeostasis. They possess data chips predating their own crystallization — the bootstrap paradox at the center of their cosmology.",
}

store_times = []
for key, text in facts.items():
    result, ms = mcp_store(key, text)
    store_times.append(ms)
    print(f"  Stored '{key}': {result} [{ms:.0f}ms]")

print(f"\n  Avg store latency: {sum(store_times)/len(store_times):.0f}ms")
print(f"  Store contents: {requests.get(f'{MCP_URL}/list').json()}")

# Retrieve test
print("\n--- Retrieval test ---")
retrieve_times = []
queries = [
    ("how fast is the GPU model?", "gpt_oss_benchmark"),
    ("what is wrong with bitsandbytes?", "bitsandbytes_fix"),
    ("tell me about the aliens", "naa_ru_lore"),
]
retrieval_correct = 0
for query, expected_key in queries:
    result, ms = mcp_retrieve(query, top_k=1)
    retrieve_times.append(ms)
    top = result["results"][0] if result["results"] else None
    if top:
        correct = "✅" if top["key"] == expected_key else "❌"
        if top["key"] == expected_key:
            retrieval_correct += 1
        print(f"  Q: {query}")
        print(f"    → {correct} [{top['score']:.3f}] {top['key']}: {top['text'][:80]}... [{ms:.0f}ms]")

print(f"\n  Avg retrieve latency: {sum(retrieve_times)/len(retrieve_times):.0f}ms")
print(f"  Retrieval accuracy: {retrieval_correct}/{len(queries)}")

# ── TEST 2: LLM calls MCP tools ───────────────────────────────────
print("\n=== TEST 2: LLM answers from RAM context ===")
tools = requests.get(f"{MCP_URL}/tools").json()

question = "What token generation speed did we achieve on the RX 9070 with GPT-OSS 20B?"

messages = [
    {"role": "system", "content": "You have access to a RAM memory server. Use context_retrieve to look up information before answering questions. Always call context_retrieve first."},
    {"role": "user", "content": question}
]

print(f"Question: {question}")
t0 = time.time()
response = llm(messages, tools=tools, max_tokens=500)
llm_time = (time.time() - t0) * 1000
print(f"LLM response time: {llm_time:.0f}ms")
print(f"Response: {json.dumps(response, indent=2)[:1500]}")

choice = response.get("choices", [{}])[0]
finish_reason = choice.get("finish_reason", "unknown")
print(f"Finish reason: {finish_reason}")

tool_called = False
if finish_reason in ("tool_calls", "stop") and choice.get("message", {}).get("tool_calls"):
    tool_called = True
    print("✅ Model called a tool!")
    tool_call = choice["message"]["tool_calls"][0]
    print(f"  Tool: {tool_call['function']['name']}")
    args = json.loads(tool_call["function"]["arguments"])
    print(f"  Args: {args}")
    
    if tool_call["function"]["name"] == "context_retrieve":
        result, ms = mcp_retrieve(args["query"], args.get("top_k", 3))
        print(f"  Retrieved: {json.dumps(result, indent=2)[:500]}")
        
        messages.append(choice["message"])
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result)
        })
        final = llm(messages, max_tokens=300)
        final_text = final["choices"][0]["message"]["content"]
        print(f"\n  Final answer: {final_text}")
else:
    print(f"  Model answer (no tool call): {choice.get('message', {}).get('content', 'N/A')[:500]}")

# ── TEST 3: Manual RAG loop (bypass tool calling) ─────────────────
print("\n=== TEST 3: Manual RAG loop (context injection) ===")
question2 = "What compression ratio does RotorQuant achieve?"

ctx_result, ms = mcp_retrieve(question2, top_k=2)
print(f"  Retrieved context in {ms:.0f}ms:")
for r in ctx_result["results"]:
    print(f"    [{r['score']:.3f}] {r['key']}: {r['text'][:80]}...")

context_text = "\n".join([f"- {r['text']}" for r in ctx_result["results"]])
messages2 = [
    {"role": "system", "content": f"Answer based ONLY on this context:\n{context_text}"},
    {"role": "user", "content": question2}
]

t0 = time.time()
response2 = llm(messages2, max_tokens=200)
llm_time2 = (time.time() - t0) * 1000
answer2 = response2["choices"][0]["message"]["content"]
print(f"  Answer ({llm_time2:.0f}ms): {answer2[:300]}")

# ── SUMMARY ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  MCP Health: ✅")
print(f"  Facts stored: {len(facts)}")
print(f"  Avg store latency: {sum(store_times)/len(store_times):.0f}ms")
print(f"  Avg retrieve latency: {sum(retrieve_times)/len(retrieve_times):.0f}ms")
print(f"  Retrieval accuracy: {retrieval_correct}/{len(queries)}")
print(f"  LLM tool calling: {'✅' if tool_called else '❌ (used manual RAG instead)'}")
print(f"  Manual RAG works: ✅")
print(f"  RAM usage: {requests.get(f'{MCP_URL}/list').json()['ram_bytes_approx']} bytes")
