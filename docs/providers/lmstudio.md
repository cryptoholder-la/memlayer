# LM Studio: Local LLM Provider

## Overview

LM Studio is a user-friendly desktop application that allows you to discover, download, and run local LLMs with a graphical interface. It provides a local server that mimics the OpenAI API, making it incredibly easy to integrate with Memlayer.

**Key Benefits:**
- ✅ **GUI Interface:** Easy to find and test models (GGUF format)
- ✅ **Hardware Optimization:** Easy GPU offloading sliders (NVIDIA, AMD, Apple Silicon)
- ✅ **Privacy:** Completely offline operation
- ✅ **Compatibility:** Drop-in replacement for OpenAI-based workflows

---

## Installation

### 1. Install LM Studio

Download and install the application for your OS (macOS, Windows, or Linux) from [lmstudio.ai](https://lmstudio.ai).

### 2. Install Memlayer

```bash
pip install memlayer openai
```
*(Note: The `openai` library is required because Memlayer communicates with LM Studio using the OpenAI protocol).*

---

## Quick Start

### 1. Prepare LM Studio
1.  Open **LM Studio**.
2.  Click the **Magnifying Glass** (Search) icon.
3.  Search for a model (e.g., `qwen 2.5 7b` or `llama 3.1 8b`).
4.  Download a quantization level that fits your RAM (e.g., `Q4_K_M` or `Q6_K`).

### 2. Start the Local Server
1.  Click the **Developer / Local Server** icon (usually `< >` on the left sidebar).
2.  Select your downloaded model from the top dropdown to load it into memory.
3.  Click the green **Start Server** button.
4.  Note the URL (Default: `http://localhost:1234`).

### 3. Basic Usage

```python
from memlayer.wrappers.lmstudio import LMStudio

# Initialize client pointing to local server
client = LMStudio(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",    # Any string works
    user_id="alice",
    operation_mode="local"  # Important: Keeps embeddings local too
)

# Chat
response = client.chat([
    {"role": "user", "content": "My name is Alice and I'm a sci-fi writer."}
])
print(response)

# Memory Recall
response = client.chat([
    {"role": "user", "content": "What is my profession?"}
])
print(response)  # "You are a sci-fi writer."
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
from memlayer import LMStudio

client = LMStudio(
    # Server Connection
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="local-model", # Usually ignored by LM Studio, but good for logging
    
    # Memory & Privacy
    user_id="alice",
    operation_mode="local", # "local", "online", or "lightweight"
    
    # Storage Paths
    storage_path="./memlayer_data",
    
    # Tuning
    temperature=0.7,
)
```

### Operation Modes

**Local Mode (Recommended for LM Studio users):**
```python
client = LMStudio(operation_mode="local")
```
*   **LLM:** Local (LM Studio)
*   **Embeddings:** Local (HuggingFace `all-MiniLM-L6-v2` running in Python)
*   **Privacy:** 100% Offline.

**Online Mode:**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

client = LMStudio(operation_mode="online")
```
*   **LLM:** Local (LM Studio)
*   **Embeddings:** OpenAI API (`text-embedding-3-small`)
*   **Privacy:** Text leaves machine for embedding, but inference is local.

---

## Streaming Support

Memlayer supports streaming with LM Studio just like any other provider.

```python
from memlayer import LMStudio

client = LMStudio(operation_mode="local")

print("Assistant: ", end="")
for chunk in client.chat([
    {"role": "user", "content": "Write a short poem about memory."}
], stream=True):
    print(chunk, end="", flush=True)
```

---

## Performance Tuning

### 1. GPU Offloading
In the LM Studio **Local Server** tab (right sidebar), look for **"GPU Offload"**.
*   **Max:** Slide to the right to put as many layers as possible on your GPU.
*   **Impact:** Drastically improves speed.

### 2. Context Window
In the LM Studio sidebar, check **Context Length**.
*   Ensure it is at least `2048` or `4096`.
*   If Memlayer memories grow large, you may need `8192` (if your model supports it).

### 3. System Prompt
In LM Studio settings:
*   **System Prompt:** Ensure this is enabled.
*   Memlayer injects instructions about "Using Tools" into the system prompt. If the model behaves unexpectedly, ensure LM Studio isn't overriding the system prompt sent by the client.

---

## Troubleshooting

### "Connection Refused" or "Target Machine Actively Refused"
**Cause:** The LM Studio server is not running.
**Solution:** Go to the "Local Server" tab in LM Studio and click the green **Start Server** button.

### "Error: 400... response_format"
**Cause:** Older versions of LM Studio or specific model loaders may not support the `json_object` enforcement used by OpenAI.
**Solution:** Memlayer's `LMStudio` wrapper handles this automatically by stripping strict JSON flags. Ensure you are using the latest version of Memlayer.

---

## Complete Example

```python
from memlayer import LMStudio
import time

# 1. Setup Client
client = LMStudio(
    base_url="http://localhost:1234/v1",
    model="qwen3-14b",
    user_id="alice_local"
)

def chat_with_memory(text):
    print(f"\n👤 User: {text}")
    print(f"🤖 AI: ", end="", flush=True)
    
    # Stream response
    full_response = ""
    for chunk in client.chat([{"role": "user", "content": text}], stream=True):
        print(chunk, end="", flush=True)
        full_response += chunk
    print()

# 2. Teach
chat_with_memory("My favorite food is sushi and I live in Tokyo.")

# 3. Wait for Background Processing
print("\n[System] Consolidating memories (wait 5s)...")
time.sleep(5)

# 4. Recall (New Session)
# We reset the message history here to prove it's fetching from DB, not context window
client.chat_history = [] 
chat_with_memory("Where do I live and what should I eat for dinner?")
```