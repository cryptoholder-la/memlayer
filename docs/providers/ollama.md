# Ollama: Local LLM Provider

## Overview

Ollama enables you to run LLMs locally on your machine, providing complete privacy and zero API costs. Memlayer's Ollama wrapper adds persistent memory capabilities to any Ollama-supported model.

**Key Benefits:**
- ✅ Fully offline operation (no internet required)
- ✅ Complete data privacy (nothing leaves your machine)
- ✅ Zero API costs
- ✅ Fast inference on modern hardware
- ✅ Support for 100+ open-source models

---

## Installation

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com/download](https://ollama.com/download)

**Verify installation:**
```bash
ollama --version
```

### 2. Install Memlayer with Ollama Support

```bash
pip install memlayer ollama
```

---

## Quick Start

### Start Ollama Server

```bash
ollama serve
```

Leave this running in a terminal. Default address: `http://localhost:11434`

### Pull a Model

```bash
ollama pull qwen3:14b

### Basic Usage

```python
from memlayer.wrappers.ollama import Ollama

# Initialize with local model
client = Ollama(
    model="qwen3:14b,
    host="http://localhost:11434",
    user_id="alice",
    operation_mode="local"  # Use local embeddings too
)

# Use like any other Memlayer client
response = client.chat([
    {"role": "user", "content": "My name is Alice and I work on Project Phoenix"}
])
print(response)

# Later - it remembers!
response = client.chat([
    {"role": "user", "content": "What project do I work on?"}
])
print(response)  # "You work on Project Phoenix"
```

---

## Recommended Models

Since Memlayer relies on **Tool Calling** and **JSON Extraction** for memory management, you must use models capable of instruction following.

### For Speed (< 2s response)
* **Gemma 3 (1B–3B, Instruct)** – Extremely fast, long context, very efficient.
* **Instella-3B (Instruct)** – New 2025 lightweight model optimized for instruction-following.
* **Mistral Small 3.1 (Efficient 24B, Instruct)** – Higher params but highly optimized for low-latency inference.

### For Quality (Standard)
* **Qwen 3 (32B, Instruct)** – Excellent logic, tool use, and long context.
* **Llama 4 (8B–70B, Scout/Maverick variants)** – The new industry standard for local models in 2025.
* **Mistral Medium 3 (~24–32B, Instruct)** – Strong balance of performance and compute cost.

### For Best Performance
* **Qwen 3 (235B-A22B hybrid)** – State-of-the-art reasoning with massive context windows.
* **Llama 4 Behemoth (Large-scale)** – High-end open model with near GPT-4.5-class capability.

---

## Configuration

### Complete Configuration Example

```python
from memlayer.wrappers.ollama import Ollama

client = Ollama(
    # Model settings
    model="qwen3:14b",
    host="http://localhost:11434",
    
    # Memory settings
    user_id="alice",
    operation_mode="local",  # Use local embeddings
    
    # Storage paths
    chroma_dir="./chroma_db",
    networkx_path="./knowledge_graph.pkl",
    
    # Performance tuning
    max_search_results=5,
    search_tier="balanced",
    salience_threshold=0.5,
    
    # Ollama-specific
    temperature=0.7,
    num_ctx=4096  # Context window size
)
```

### Operation Modes with Ollama

**Local mode (recommended):**
```python
client = Ollama(
    model="qwen3:14b",
    operation_mode="local"  # Local embeddings, fully offline
)
# First call: ~5-10s (loads sentence-transformer model)
# Subsequent calls: fast
```

**Online mode (hybrid):**
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"

client = Ollama(
    model="qwen3:14b",
    operation_mode="online"  # LLM local, embeddings via OpenAI API
)
# Faster startup, but requires internet for embeddings
```

**Lightweight mode (fastest startup):**
```python
client = Ollama(
    model="qwen3:14b",
    operation_mode="lightweight"  # No embeddings, graph-only
)
# Instant startup, keyword-based search only
```

---

## Streaming Support

Ollama fully supports streaming responses:

```python
from memlayer.wrappers.ollama import Ollama

client = Ollama(model="qwen3:14b", operation_mode="local")

# Stream response chunks
for chunk in client.chat([
    {"role": "user", "content": "Tell me about quantum computing"}
], stream=True):
    print(chunk, end="", flush=True)
print()  # Newline after completion
```

**Performance:**
- First chunk: ~1-2s (includes memory search if needed)
- Chunks: 1-5 characters each (smooth streaming)
- Knowledge extraction: background, doesn't block stream

---

## Complete Offline Setup

Run Memlayer entirely offline with Ollama:

```python
from memlayer.wrappers.ollama import Ollama

# Fully offline - no internet required
client = Ollama(
    model="qwen3:14b",
    host="http://localhost:11434",
    operation_mode="local",  # Local sentence-transformer for embeddings
    user_id="alice"
)

# Everything runs locally:
# - LLM inference (Ollama)
# - Embeddings (sentence-transformers)
# - Vector search (ChromaDB)
# - Graph storage (NetworkX)
```

**First-time setup:**
```bash
# Pull model (one-time, requires internet)
ollama pull qwen3:14b

# First Python call downloads embedding model (one-time)
# Model: all-MiniLM-L6-v2 (~80MB)
```

**After setup:** Completely offline, no internet needed!

---

## Advanced Configuration

### Custom Ollama Host

```python
# Remote Ollama server
client = Ollama(
    model="qwen3:14b",
    host="http://192.168.1.100:11434",  # Remote server
    operation_mode="local"
)
```

### Custom Context Window

```python
client = Ollama(
    model="qwen3:14b",
    num_ctx=8192,  # Increase context window (if model supports it)
)
```

### Custom Temperature

```python
client = Ollama(
    model="qwen3:14b",
    temperature=0.3,  # Lower = more focused, higher = more creative
)
```

### Custom Embedding Model

```python
client = Ollama(
    model="qwen3:14b",
    operation_mode="local",
    embedding_model="all-mpnet-base-v2"  # Better quality, slower
)
```

---

## Performance Tuning

### Hardware Recommendations

| Model Size | RAM | GPU VRAM | Response Time |
|------------|-----|----------|---------------|
| 3B (llama3.2) | 8GB | Optional | 1-2s |
| 7B (mistral) | 16GB | Optional | 2-5s |
| 8B (llama3.1) | 16GB | 8GB+ | 2-5s |
| 70B (llama3.1:70b) | 40GB+ | 24GB+ | 5-15s |

### GPU Acceleration

Ollama automatically uses GPU if available (NVIDIA, AMD, Apple Silicon):

```bash
# Verify GPU usage
ollama run llama3.2

# In another terminal:
nvidia-smi  # For NVIDIA GPUs
# or
rocm-smi   # For AMD GPUs
```

### Model Loading Time

First inference loads model to memory (~2-5s). Keep Ollama running to avoid reload:

```bash
# Keep model loaded
ollama run llama3.2

# In another terminal/notebook, use Memlayer
# Model is already in memory, responses are instant
```

### Concurrent Requests

Ollama handles concurrent requests efficiently:

```python
import concurrent.futures

clients = [Ollama(model="llama3.2", user_id=f"user{i}") 
           for i in range(5)]

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(c.chat, [{"role": "user", "content": f"Hello {i}"}])
        for i, c in enumerate(clients)
    ]
    responses = [f.result() for f in futures]
```

---

## Troubleshooting

### "Connection refused" Error

**Problem:** Ollama server not running

**Solution:**
```bash
ollama serve
```

### Slow First Response

**Problem:** Model loading into memory

**Solution:** Keep Ollama server running with model loaded:
```bash
ollama run llama3.2
# Keep this terminal open
```

### Out of Memory

**Problem:** Model too large for your hardware

**Solution:** Use smaller model:
```bash
ollama pull llama3.2  # 3B model, needs only 8GB RAM
```

### Model Download Fails

**Problem:** Network issues during pull

**Solution:** Retry with resume:
```bash
ollama pull llama3.2  # Automatically resumes
```

### Embeddings Download Fails (Local Mode)

**Problem:** First-time sentence-transformer download fails

**Solution:** Manually download:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
# Now Memlayer will find cached model
```

---

## Complete Example

```python
from memlayer.wrappers.ollama import Ollama
import time

# Initialize fully offline client
client = Ollama(
    model="qwen3:14b,
    host="http://localhost:11434",
    operation_mode="local",
    user_id="alice"
)

def chat(message):
    """Send a message and stream the response."""
    print(f"\n🤖 Assistant: ", end="", flush=True)
    start = time.time()
    
    for chunk in client.chat([
        {"role": "user", "content": message}
    ], stream=True):
        print(chunk, end="", flush=True)
    
    elapsed = time.time() - start
    print(f"\n⏱️  Response time: {elapsed:.2f}s\n")

# Example conversation
print("👤 User: My name is Alice and I love hiking")
chat("My name is Alice and I love hiking")

print("👤 User: What do I like to do?")
chat("What do I like to do?")

print("👤 User: Plan a weekend activity for me")
chat("Plan a weekend activity for me")
```

---


## Next Steps

- **[Basics Quickstart](../basics/quickstart.md)**: General getting started guide
- **[Streaming Mode](../basics/streaming.md)**: Learn about streaming responses
- **[Operation Modes](../tuning/operation_mode.md)**: Deep dive into local vs online modes
- **[Examples](../../examples/05_providers/ollama_example.py)**: Complete working code
- **[Ollama Docs](https://github.com/ollama/ollama)**: Official Ollama documentation