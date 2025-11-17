# MemLayer

**The plug-and-play memory layer for smart, contextual agents**

MemLayer adds persistent, intelligent memory to any LLM, enabling agents that recall context across conversations, extract structured knowledge, and surface relevant information when it matters.

**<100ms Fast Search • Noise-Aware Memory Gate • Multi-Tier Retrieval Modes • 100% Local • Zero Config**

---

## Quick start

Install:

```bash
pip install memlayer
```

Basic usage:

```python
from memlayer.wrappers.openai import OpenAI

client = OpenAI(model="gpt-4.1-mini", storage_path="./memories", user_id="user_123")
client.chat([{"role": "user", "content": "My name is Alice and I work at TechCorp"}])
response = client.chat([{"role": "user", "content": "Where do I work?"}])
# -> "You work at TechCorp."
```

MemLayer automatically filters, extracts, stores and retrieves relevant memories, no manual prompts required.

---

## What makes MemLayer different

- **Selective long-term memory** — we only store what matters, not every chat line.  
- **Hybrid storage** — semantic vectors for recall plus a lightweight knowledge graph for relationships and updates.  
- **Noise-aware gate** — cheap salience checks (prototype embeddings + TF-IDF) keep the store clean.  
- **Multi-tier retrieval** — Fast / Balanced / Deep modes so you can trade latency for depth.  
- **Offline-first** — runs locally with no external services required.  
- **Zero config** — drop it into your app and start remembering.

---

## Explore the docs

- **[Basics & Quickstart](basics/overview.md)** — architecture and getting started
- **[Search Tiers & Modes](operation_modes.md)** — how retrieval and salience modes work
- **[API Reference](API_REFERENCE.md)** — full method docs and examples
- **[Providers](providers/README.md)** — OpenAI, Claude, Gemini, Ollama, and local model tips
- **[Storage Backends](storage/chroma.md)** — ChromaDB + NetworkX details

Or open the sidebar to browse everything.

---

## Examples & development

See `examples/` for runnable demos (basics, search tiers, provider samples).  
Clone and run locally:

```bash
git clone https://github.com/divagr18/memlayer.git
cd memlayer
pip install -e .
python examples/01_basics/getting_started.py
```

---

## Contributing

Bug reports, PRs, and documentation fixes are welcome. See the repo `CONTRIBUTING.md` for guidelines.

---

If you want a slightly shorter or more technical landing page (or want the badges moved here), tell me which tone you prefer and I’ll adapt it.
