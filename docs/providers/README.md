# Provider-Specific Documentation

Memlayer supports multiple LLM providers with a unified API. Each provider has specific configuration requirements and features documented here.

## Supported Providers

### [OpenAI](./openai.md)
- **Models**: GPT-4.1, GPT 5 etc
- **Streaming**: ✅ Full support
- **Best for**: Production applications, fastest API responses
- **Setup**: Requires `OPENAI_API_KEY` environment variable

### [Anthropic Claude](./claude.md)
- **Models**: Claude 4.5 Sonnet, Claude 4 Opus, Claude 4 Haiku
- **Streaming**: ✅ Full support
- **Best for**: Long conversations, complex reasoning
- **Setup**: Requires `ANTHROPIC_API_KEY` environment variable

### [Google Gemini](./gemini.md)
- **Models**: Gemini 2.5 Flash, Gemini 2.5 Pro
- **Streaming**: ✅ Full support
- **Best for**: Multimodal applications, cost efficiency
- **Setup**: Requires `GOOGLE_API_KEY` environment variable

### [Ollama (Local Models)](./ollama.md) 
- **Models**: Llama 3.2, Llama 3.1, Mistral, Phi 3, 100+ more
- **Streaming**: ✅ Full support
- **Best for**: Privacy, offline use, zero API costs
- **Setup**: Requires local Ollama server (`ollama serve`)

## Quick Comparison

| Provider | API Cost | Latency | Privacy | Offline |
|----------|----------|---------|---------|---------|
| OpenAI | $$ | Fast | Cloud | ❌ |
| Claude | $$ | Fast | Cloud | ❌ |
| Gemini | $ | Fast | Cloud | ❌ |
| Ollama | Free | Medium | Local | ✅ |

## Configuration Basics

All providers share the same Memlayer API:

```python
from memlayer.wrappers.openai import OpenAI
from memlayer.wrappers.claude import Claude
from memlayer.wrappers.gemini import Gemini
from memlayer.wrappers.ollama import Ollama

# OpenAI
client = OpenAI(
    api_key="your-key",
    model="gpt-4.1-mini",
    user_id="alice"
)

# Claude
client = Claude(
    api_key="your-key",
    model="claude-3-5-sonnet-20241022",
    user_id="alice"
)

# Gemini
client = Gemini(
    api_key="your-key",
    model="gemini-2.5-flash",
    user_id="alice"
)

# Ollama (local)
client = Ollama(
    model="llama3.2",
    host="http://localhost:11434",
    user_id="alice",
    operation_mode="local"  # Fully offline
)
```

## Common Features Across All Providers

### Memory & Knowledge Graph
All providers support:
- ✅ Automatic knowledge extraction
- ✅ Persistent memory across sessions
- ✅ Hybrid search (vector + graph)
- ✅ Time-aware facts with expiration
- ✅ User-isolated memory spaces

### Streaming Responses
All providers support streaming:
```python
for chunk in client.chat([
    {"role": "user", "content": "Tell me a story"}
], stream=True):
    print(chunk, end="", flush=True)
```

### Operation Modes
All providers support three operation modes:
- **online**: API-based embeddings (fast startup)
- **local**: Local embeddings (privacy, offline)
- **lightweight**: No embeddings (instant startup)

## Provider-Specific Pages

Click on any provider below for detailed setup instructions:

- **[openai.md](./openai.md)** — OpenAI configuration, models, and tips
- **[claude.md](./claude.md)** — Anthropic Claude setup and features
- **[gemini.md](./gemini.md)** — Google Gemini configuration
- **[ollama.md](./ollama.md)** — **🆕 Complete guide to local models**: installation, model recommendations, fully offline setup

## Getting Started

1. **Choose a provider** based on your needs (cost, privacy, performance)
2. **Set up credentials** (see individual provider pages)
3. **Follow the quickstart** — [docs/basics/quickstart.md](../basics/quickstart.md)
4. **Enable streaming** (optional) — [docs/basics/streaming.md](../basics/streaming.md)

## Related Documentation

- **[Basics Overview](../basics/overview.md)**: How Memlayer works
- **[Quickstart Guide](../basics/quickstart.md)**: Get started in 5 minutes
- **[Streaming Mode](../basics/streaming.md)**: Stream responses from any provider
- **[Operation Modes](../tuning/operation_mode.md)**: Choose online, local, or lightweight mode
- **[Examples](../../examples/05_providers/README.md)**: Working code for each provider
