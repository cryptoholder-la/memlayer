# Memlayer API Reference

Complete reference for all public methods and their parameters.

---

## Client Initialization

All wrapper classes (`OpenAI`, `Claude`, `Gemini`, `Ollama`) share these common initialization parameters:

```python
from memlayer.wrappers.openai import OpenAI
from memlayer.wrappers.claude import Claude
from memlayer.wrappers.gemini import Gemini
from memlayer.wrappers.ollama import Ollama
```

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | API key (read from env if not provided) |
| `model` | `str` | *varies* | Model name/identifier |
| `user_id` | `str` | `"default_user"` | User identifier for memory isolation |
| `operation_mode` | `str` | `"online"` | `"online"`, `"local"`, or `"lightweight"` |
| `chroma_dir` | `str` | `"./chroma_db"` | Path to ChromaDB storage directory |
| `networkx_path` | `str` | `"./knowledge_graph.pkl"` | Path to NetworkX graph file |
| `salience_threshold` | `float` | `0.5` | Threshold for content filtering (0.0-1.0) |
| `embedding_model` | `str` | *varies* | Embedding model name |
| `max_search_results` | `int` | `5` | Maximum search results to return |
| `search_tier` | `str` | `"balanced"` | `"fast"`, `"balanced"`, or `"deep"` |
| `curation_interval` | `int` | `3600` | Curation check interval in seconds |
| `temperature` | `float` | `0.7` | LLM temperature |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |

### Provider-Specific Parameters

**OpenAI:**
```python
client = OpenAI(
    api_key="sk-...",
    model="gpt-4.1-mini",
    user_id="alice"
)
```

**Claude:**
```python
client = Claude(
    api_key="sk-ant-...",
    model="claude-4-sonnet",
    user_id="alice"
)
```

**Gemini:**
```python
client = Gemini(
    api_key="AIza...",
    model="gemini-2.5-flash",
    user_id="alice"
)
```

**Ollama:**
```python
client = Ollama(
    model="llama3.2",
    host="http://localhost:11434",  # Additional parameter
    user_id="alice",
    operation_mode="local"
)
```

---

## Core Methods

### `chat()`

Send a chat completion request with memory capabilities.

**Signature:**
```python
def chat(
    messages: List[Dict[str, str]],
    stream: bool = False,
    **kwargs
) -> str | Generator[str, None, None]
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | `List[Dict[str, str]]` | ✅ | List of message dicts with `role` and `content` |
| `stream` | `bool` | ❌ | If `True`, returns generator yielding chunks |
| `**kwargs` | `Any` | ❌ | Additional provider-specific arguments |

**Returns:**
- If `stream=False`: `str` - Complete response text
- If `stream=True`: `Generator[str, None, None]` - Generator yielding response chunks

**Example:**
```python
# Non-streaming
response = client.chat([
    {"role": "user", "content": "Hello!"}
])

# Streaming
for chunk in client.chat([
    {"role": "user", "content": "Hello!"}
], stream=True):
    print(chunk, end="", flush=True)
```

---

### `update_from_text()`

Directly ingest text into memory without conversational interaction.

**Signature:**
```python
def update_from_text(text_block: str) -> None
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text_block` | `str` | ✅ | Text content to analyze and store |

**Returns:** `None` (runs asynchronously in background)

**Example:**
```python
client.update_from_text("""
Meeting notes from Nov 15:
- Q4 deadline: December 20th
- New team member: Bob
- Budget increased 15%
""")
```

---

### `synthesize_answer()`

Generate a memory-grounded answer to a specific question.

**Signature:**
```python
def synthesize_answer(
    question: str,
    return_object: bool = False
) -> str | AnswerObject
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | `str` | ✅ | Question to answer |
| `return_object` | `bool` | ❌ | If `True`, returns detailed `AnswerObject` |

**Returns:**
- If `return_object=False`: `str` - Answer text
- If `return_object=True`: `AnswerObject` with fields:
  - `answer: str` - The synthesized answer
  - `sources: List[str]` - Source facts used
  - `confidence: float` - Confidence score (0.0-1.0)

**Example:**
```python
# Simple answer
answer = client.synthesize_answer("What is the Q4 deadline?")

# Detailed answer with sources
answer_obj = client.synthesize_answer(
    "What is the Q4 deadline?",
    return_object=True
)
print(f"Answer: {answer_obj.answer}")
print(f"Sources: {answer_obj.sources}")
print(f"Confidence: {answer_obj.confidence}")
```

---

### `analyze_and_extract_knowledge()`

Extract structured knowledge from text (internal method, but can be called directly).

**Signature:**
```python
def analyze_and_extract_knowledge(text: str) -> Dict[str, List[Dict]]
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | `str` | ✅ | Text to analyze |

**Returns:** `Dict` with keys:
- `facts`: `List[Dict]` - Extracted facts with `fact`, `importance_score`, `expiration_date`
- `entities`: `List[Dict]` - Entities with `name` and `type`
- `relationships`: `List[Dict]` - Relationships with `subject`, `predicate`, `object`

**Example:**
```python
knowledge = client.analyze_and_extract_knowledge(
    "Alice works at TechCorp as a Senior Engineer"
)

print(knowledge["facts"])
# [{"fact": "Alice works at TechCorp", "importance_score": 0.9, ...}]

print(knowledge["entities"])
# [{"name": "Alice", "type": "Person"}, {"name": "TechCorp", "type": "Organization"}]

print(knowledge["relationships"])
# [{"subject": "Alice", "predicate": "works at", "object": "TechCorp"}]
```

---

## Search Service

Access via `client.search_service`.

### `search()`

Search the knowledge graph and vector store.

**Signature:**
```python
def search(
    query: str,
    user_id: str,
    search_tier: str = "balanced",
    llm_client: Optional[Any] = None
) -> Dict[str, Any]
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `str` | ✅ | Search query |
| `user_id` | `str` | ✅ | User ID for memory isolation |
| `search_tier` | `str` | ❌ | `"fast"`, `"balanced"`, or `"deep"` |
| `llm_client` | `Any` | ❌ | LLM client for entity extraction in deep search |

**Returns:** `Dict` with keys:
- `result`: `str` - Formatted search results
- `trace`: `Trace` - Observability trace object

**Example:**
```python
search_output = client.search_service.search(
    query="What do I know about Alice?",
    user_id="user123",
    search_tier="deep",
    llm_client=client
)

print(search_output["result"])
print(f"Search took: {search_output['trace'].total_time_ms}ms")
```

---

## Consolidation Service

Access via `client.consolidation_service`.

### `consolidate()`

Extract and store knowledge from text (runs in background thread).

**Signature:**
```python
def consolidate(text: str, user_id: str) -> None
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | `str` | ✅ | Text to consolidate |
| `user_id` | `str` | ✅ | User ID for memory isolation |

**Returns:** `None` (runs asynchronously)

**Example:**
```python
client.consolidation_service.consolidate(
    "Alice mentioned the project deadline is next Friday",
    user_id="user123"
)
```

---

## Graph Storage

Access via `client.graph_storage`.

### `add_task()`

Schedule a task with a due date.

**Signature:**
```python
def add_task(
    description: str,
    due_timestamp: float,
    user_id: str
) -> str
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `description` | `str` | ✅ | Task description |
| `due_timestamp` | `float` | ✅ | Unix timestamp when task is due |
| `user_id` | `str` | ✅ | User ID for task ownership |

**Returns:** `str` - Task ID

**Example:**
```python
from datetime import datetime, timedelta

due_time = (datetime.now() + timedelta(days=7)).timestamp()
task_id = client.graph_storage.add_task(
    description="Review project proposal",
    due_timestamp=due_time,
    user_id="alice"
)
```

### `get_entity_subgraph()`

Get entity and its relationships from the graph.

**Signature:**
```python
def get_entity_subgraph(
    entity_name: str,
    user_id: str,
    max_depth: int = 2
) -> Dict[str, Any]
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity_name` | `str` | ✅ | Entity name to query |
| `user_id` | `str` | ✅ | User ID for isolation |
| `max_depth` | `int` | ❌ | Maximum traversal depth |

**Returns:** `Dict` with entity data and relationships

**Example:**
```python
subgraph = client.graph_storage.get_entity_subgraph(
    entity_name="Alice",
    user_id="user123",
    max_depth=2
)
```

---

## Vector Storage (ChromaDB)

Access via `client.chroma_storage`.

### `add_facts()`

Add facts to vector store with embeddings.

**Signature:**
```python
def add_facts(
    facts: List[str],
    metadata_list: List[Dict],
    user_id: str
) -> None
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `facts` | `List[str]` | ✅ | List of fact strings |
| `metadata_list` | `List[Dict]` | ✅ | Metadata for each fact |
| `user_id` | `str` | ✅ | User ID for isolation |

**Returns:** `None`

### `search_facts()`

Search for similar facts using vector similarity.

**Signature:**
```python
def search_facts(
    query: str,
    user_id: str,
    n_results: int = 5
) -> Dict[str, Any]
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | `str` | ✅ | Search query |
| `user_id` | `str` | ✅ | User ID for isolation |
| `n_results` | `int` | ❌ | Maximum results to return |

**Returns:** `Dict` with search results

---

## Observability

### Trace Object

Returned by search operations for performance monitoring.

**Attributes:**
- `total_time_ms`: `float` - Total search time in milliseconds
- `vector_search_time_ms`: `float` - Time spent in vector search
- `graph_traversal_time_ms`: `float` - Time spent in graph traversal
- `entity_extraction_time_ms`: `float` - Time spent extracting entities
- `search_tier`: `str` - Search tier used
- `results_count`: `int` - Number of results returned

**Example:**
```python
output = client.search_service.search(query="...", user_id="...")
trace = output["trace"]

print(f"Total: {trace.total_time_ms}ms")
print(f"Vector search: {trace.vector_search_time_ms}ms")
print(f"Graph traversal: {trace.graph_traversal_time_ms}ms")
```

---

## Message Format

All `chat()` methods expect messages in this format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},  # Optional
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]
```

**Roles:**
- `"system"`: System instructions (optional)
- `"user"`: User messages
- `"assistant"`: Assistant responses

---

## Tool Schema

Memlayer automatically provides these tools to the LLM:

### `search_memory`

**Parameters:**
- `query` (string): What to search for
- `search_tier` (string): `"fast"`, `"balanced"`, or `"deep"`

### `schedule_task`

**Parameters:**
- `task_description` (string): Task description
- `due_date` (string): ISO 8601 date string

The LLM calls these tools automatically when needed - no manual configuration required.

---

## Configuration Classes

### `MemlayerConfig`

Central configuration object (advanced usage).

```python
from memlayer.config import MemlayerConfig

config = MemlayerConfig(
    operation_mode="online",
    salience_threshold=0.5,
    embedding_model="text-embedding-3-small"
)
```

**Key Attributes:**
- `operation_mode`: `str`
- `salience_threshold`: `float`
- `embedding_model`: `str`
- `chroma_dir`: `str`
- `networkx_path`: `str`
- `max_search_results`: `int`
- `search_tier`: `str`

---

## Error Handling

All methods may raise:
- `ConnectionError`: API connection issues
- `ValueError`: Invalid parameters
- `FileNotFoundError`: Storage path issues
- Provider-specific exceptions (OpenAI, Anthropic, Google errors)

**Example:**
```python
try:
    response = client.chat(messages)
except ConnectionError as e:
    print(f"API connection failed: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

---

## Type Hints

```python
from typing import List, Dict, Generator, Optional, Any

# Message format
Message = Dict[str, str]  # {"role": str, "content": str}

# Chat signatures
def chat(
    messages: List[Message],
    stream: bool = False,
    **kwargs: Any
) -> str | Generator[str, None, None]: ...

# Knowledge format
Knowledge = Dict[str, List[Dict[str, Any]]]  # {"facts": [...], "entities": [...], "relationships": [...]}
```

---

## Quick Reference

| Method | Purpose | Returns |
|--------|---------|---------|
| `chat(messages, stream=False)` | Send chat message | `str` or `Generator` |
| `update_from_text(text)` | Import knowledge | `None` |
| `synthesize_answer(question)` | Memory-grounded Q&A | `str` or `AnswerObject` |
| `search_service.search(query)` | Search memory | `Dict[str, Any]` |
| `graph_storage.add_task(desc, time)` | Schedule task | `str` (task_id) |
| `analyze_and_extract_knowledge(text)` | Extract structured data | `Dict[str, List]` |

---

## See Also

- **[Overview](basics/overview.md)**: Architecture and concepts
- **[Quickstart](basics/quickstart.md)**: Getting started guide
- **[Streaming](basics/streaming.md)**: Streaming mode details
- **[Examples](../examples/README.md)**: Working code examples
