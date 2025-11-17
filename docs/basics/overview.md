# Memlayer Overview

## What is Memlayer?

Memlayer is a memory-enhanced LLM wrapper that automatically builds and maintains a persistent knowledge graph from your conversations. It adds memory capabilities to any LLM provider (OpenAI, Claude, Gemini, Ollama) without changing how you interact with them.

## Core Architecture

### How It Works

1. **Chat Flow**: When you send a message via `.chat()`, Memlayer:
   - Searches the knowledge graph for relevant context
   - Injects that context into the LLM prompt via tool calls
   - Returns the LLM's response to you
   - Asynchronously extracts knowledge and updates the graph

2. **Knowledge Extraction**: After each conversation turn:
   - Text is analyzed by a fast model (background thread)
   - Facts, entities, and relationships are extracted
   - Salience gate filters out trivial information
   - Knowledge is stored in both vector DB and graph DB

3. **Memory Search**: When the LLM needs context:
   - Hybrid search combines vector similarity + graph traversal
   - Three search tiers available: `fast`, `balanced`, `deep`
   - Results are ranked and returned as context

4. **Background Services**:
   - **Consolidation**: Extracts knowledge from conversations (async)
   - **Curation**: Expires time-sensitive facts (background thread)
   - **Salience Gate**: Filters low-value information before storage

## Data Flow

### Normal Chat (Non-Streaming)

```
User Message
    │
    ▼
Memory Search (if LLM calls tool)
    │
    ▼
LLM Response Generated
    │
    ├─► Return to User
    │
    └─► Background: Extract Knowledge → Store in Graph
```

### Streaming Chat

```
User Message
    │
    ▼
Memory Search (if LLM calls tool)
    │
    ▼
LLM Starts Streaming
    │
    ├─► Yield chunks to user in real-time
    │
    └─► Background: Buffer full response
            │
            └─► After stream completes → Extract Knowledge → Store in Graph
```

## Key Features

### 1. Provider-Agnostic
Works with OpenAI, Anthropic Claude, Google Gemini, and local Ollama models. Same API across all providers.

### 2. Automatic Memory Tools
LLM automatically gets access to:
- `search_memory`: Hybrid vector + graph search
- `schedule_task`: Create time-based reminders

### 3. Flexible Search Tiers
- **fast**: Vector-only search, <100ms
- **balanced**: Vector + 1-hop graph traversal
- **deep**: Full graph traversal with entity extraction

### 4. Knowledge Graph Features
- Entity deduplication (e.g., "John" = "John Smith")
- Relationship tracking between entities
- Time-aware facts with expiration dates
- Importance scoring for fact prioritization

### 5. Operation Modes
Choose embedding strategy based on your needs:
- **online**: API-based embeddings (OpenAI), fast startup
- **local**: Local sentence-transformer model, no API costs
- **lightweight**: Graph-only, no embeddings, fastest startup

## Configuration Options

```python
from memlayer.wrappers.openai import OpenAI

client = OpenAI(
    # Core settings
    api_key="your-key",
    model="gpt-4.1-mini",
    user_id="user123",
    
    # Memory behavior
    operation_mode="online",        # online | local | lightweight
    salience_threshold=0.5,         # 0.0-1.0, filters trivial content
    
    # Storage paths
    chroma_dir="./my_chroma_db",
    networkx_path="./my_graph.pkl",
    
    # Search behavior
    max_search_results=5,
    search_tier="balanced",          # fast | balanced | deep
    
    # Performance tuning
    curation_interval=3600,          # Check for expired facts every hour
    embedding_model="text-embedding-3-small"
)
```

## Common Usage Patterns

### Basic Chat
```python
response = client.chat([
    {"role": "user", "content": "My name is Alice"}
])
# Knowledge automatically extracted and stored
```

### Streaming Chat
```python
for chunk in client.chat(
    [{"role": "user", "content": "What's my name?"}],
    stream=True
):
    print(chunk, end="", flush=True)
```

### Direct Knowledge Ingestion
```python
# Import knowledge from documents
client.update_from_text("""
Project Phoenix is led by Alice.
The project deadline is December 1st.
""")
```

### Synthesized Q&A
```python
# Get memory-grounded answer
answer = client.synthesize_answer("Who leads Project Phoenix?")
```

## Performance Characteristics

| Component | Latency | Notes |
|-----------|---------|-------|
| Memory search (fast) | 50-100ms | Vector search only |
| Memory search (balanced) | 100-300ms | Vector + 1-hop graph |
| Memory search (deep) | 300-1000ms | Full graph traversal |
| Knowledge extraction | 1-3s | Background, doesn't block response |
| Consolidation | 1-2s | Async, uses fast model |
| First-time salience init | 1-2s | Cached after first run |

## Best Practices

1. **Choose the right operation mode**:
   - Serverless → `online` mode
   - Privacy-sensitive → `local` mode
   - Demos/prototypes → `lightweight` mode

2. **Use streaming for better UX**:
   - First chunk arrives in 1-3s
   - Knowledge extraction happens in background
   - User sees response immediately

3. **Tune salience threshold**:
   - Low (0.3-0.5): Keep more memories, higher storage
   - Medium (0.5-0.7): Balanced, recommended default
   - High (0.7-0.9): Only important facts, minimal storage

4. **Set expiration dates for time-sensitive facts**:
   - System automatically extracts expiration dates from text
   - Curation service removes expired facts periodically

5. **Use appropriate search tier**:
   - `fast`: Quick lookups, high-traffic applications
   - `balanced`: Default, good recall with reasonable latency
   - `deep`: Complex questions needing graph reasoning

## Next Steps

- **[Quickstart Guide](./quickstart.md)**: Get up and running in 5 minutes
- **[Streaming Mode](./streaming.md)**: Deep dive into streaming behavior
- **[Operation Modes](../tuning/operation_mode.md)**: Architecture implications of each mode
- **[Provider Setup](../providers/README.md)**: Provider-specific configuration
