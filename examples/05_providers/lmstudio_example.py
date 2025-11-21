import time
from memlayer import LMStudio

print("="*70)
print("Memlayer - LM STUDIO (LOCAL) EXAMPLE")
print("="*70)

client = LMStudio(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3-4b-2507", # Ensure this matches your loaded model
    temperature=0.7,
    storage_path="./lmstudio_memories",
    user_id="demo_user_lmstudio",
    operation_mode="local",
)

print("\n📝 Conversation 1: Teaching the local LLM about yourself")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "Hello! My name is Sarah and I'm a sci-fi author."}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "I am currently writing a book about a Dyson Sphere civilization."}
])
print(f"\nAssistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "I struggle with writer's block when plotting chapter endings."}
])
print(f"\nAssistant: {response}")

print("\n⏳ Waiting for memory consolidation (8s)...")
time.sleep(8) 

print("\n🔍 Conversation 2: Testing memory recall (NEW SESSION)")
print("-" * 70)

# KEY FIX: We start a NEW conversation list here.
# This forces the model to look into Long-Term Memory (Tools) 
# because the info is not in the immediate context window.
recall_response = client.chat(messages=[
    {"role": "user", "content": "What is my book about?"}
])
print(f"Assistant: {recall_response}")

print("\n📊 Observability: Inspecting the last search")
print("-" * 70)

if client.last_trace:
    print(f"Search Trace:")
    print(f"  Total Duration: {client.last_trace.total_duration_ms:.1f}ms")
    if client.last_trace.metadata.get("results_found"):
        print(f"  Memories found: {client.last_trace.metadata['results_found']}")
else:
    print("No search trace available. (Model failed to use tool)")

client.close()