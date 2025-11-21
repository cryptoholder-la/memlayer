from typing import Any, Dict, List, Optional, TYPE_CHECKING
import openai
import json
import re
import os
from ..config import is_debug_mode

# Use TYPE_CHECKING to avoid slow imports at module load time
if TYPE_CHECKING:
    from ..ml_gate import SalienceGate
    from ..storage.chroma import ChromaStorage
    from ..storage.networkx import NetworkXStorage
    from ..storage.memgraph import MemgraphStorage
    from ..services import SearchService, ConsolidationService, CurationService
    from ..embedding_models import BaseEmbeddingModel, LocalEmbeddingModel
    from ..observability import Trace
    from .base import BaseLLMWrapper
else:
    SalienceGate = None
    ChromaStorage = None
    NetworkXStorage = None
    MemgraphStorage = None
    SearchService = None
    ConsolidationService = None
    BaseEmbeddingModel = None
    LocalEmbeddingModel = None
    CurationService = None
    Trace = None
    BaseLLMWrapper = object


class LMStudio(BaseLLMWrapper):
    """
    A memory-enhanced client for LM Studio (via OpenAI-compatible local server).
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "local-model",
        temperature: float = 0.7,
        storage_path: str = "./memlayer_data",
        user_id: str = "default_user",
        embedding_model: Optional["BaseEmbeddingModel"] = None,
        salience_threshold: float = 0.0,
        operation_mode: str = "local",
        scheduler_interval_seconds: int = 60,
        curation_interval_seconds: int = 3600,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.user_id = user_id
        self.storage_path = storage_path
        self.salience_threshold = salience_threshold
        self.operation_mode = operation_mode
        self._provided_embedding_model = embedding_model
        self.scheduler_interval_seconds = scheduler_interval_seconds
        self.curation_interval_seconds = curation_interval_seconds
        
        # Lazy-loaded attributes
        self._embedding_model = None
        self._vector_storage = None
        self._graph_storage = None
        self._salience_gate = None
        self._search_service = None
        self._consolidation_service = None
        self._curation_service = None
        self.last_trace: Optional["Trace"] = None

        # Initialize OpenAI client pointing to Local Server
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )
        
        import atexit
        atexit.register(self.close)

        # Tool Definitions
        self.tool_schema = [{
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Searches the user's long-term memory. REQUIRED: Use this tool whenever the user asks about themselves, past conversations, preferences, or established facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant memories."
                        },
                        "search_tier": {
                            "type": "string",
                            "enum": ["fast", "balanced", "deep"],
                            "description": "The depth of search: 'fast' for simple lookups, 'balanced' for standard questions, 'deep' for complex reasoning."
                        }
                    },
                    "required": ["query", "search_tier"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_task",
                "description": "Schedules a task or reminder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Description of the task."
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date in ISO 8601 format."
                        }
                    },
                    "required": ["task_description", "due_date"]
                }
            }
        }]

    @property
    def curation_service(self) -> "CurationService":
        if self._curation_service is None:
            from ..services import CurationService
            self._curation_service = CurationService(
                self.vector_storage, 
                self.graph_storage,
                interval_seconds=self.curation_interval_seconds
            )
            self._curation_service.start()
        return self._curation_service

    @property
    def embedding_model(self) -> "BaseEmbeddingModel":
        if self.operation_mode == "lightweight":
            return None
        if self._embedding_model is None:
            if self._provided_embedding_model is None:
                if self.operation_mode == "online":
                    from ..embedding_models import OpenAIEmbeddingModel
                    import os
                    self._embedding_model = OpenAIEmbeddingModel(
                        client=openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                        model_name="text-embedding-3-small"
                    )
                else:
                    from ..embedding_models import LocalEmbeddingModel
                    self._embedding_model = LocalEmbeddingModel()
            else:
                self._embedding_model = self._provided_embedding_model
        return self._embedding_model
    
    @property
    def vector_storage(self) -> "ChromaStorage":
        if self.operation_mode == "lightweight":
            return None
        if self._vector_storage is None:
            from ..storage.chroma import ChromaStorage
            self._vector_storage = ChromaStorage(self.storage_path, dimension=self.embedding_model.dimension)
        return self._vector_storage
    
    @property
    def graph_storage(self) -> "NetworkXStorage":
        if self._graph_storage is None:
            from ..storage.networkx import NetworkXStorage
            self._graph_storage = NetworkXStorage(self.storage_path)
        return self._graph_storage
    
    @property
    def salience_gate(self) -> "SalienceGate":
        if self._salience_gate is None:
            from ..ml_gate import SalienceGate, SalienceMode
            mode = SalienceMode(self.operation_mode.lower())
            openai_key = os.getenv("OPENAI_API_KEY") if mode == SalienceMode.ONLINE else None
            self._salience_gate = SalienceGate(
                threshold=self.salience_threshold,
                embedding_model=self.embedding_model if mode == SalienceMode.LOCAL else None,
                mode=mode,
                openai_api_key=openai_key
            )
        return self._salience_gate
    
    @property
    def search_service(self) -> "SearchService":
        if self._search_service is None:
            from ..services import SearchService
            self._search_service = SearchService(self.vector_storage, self.graph_storage, self.embedding_model)
        return self._search_service
    
    @property
    def consolidation_service(self) -> "ConsolidationService":
        if self._consolidation_service is None:
            from ..services import ConsolidationService
            self._consolidation_service = ConsolidationService(
                self.vector_storage,
                self.graph_storage,
                self.embedding_model,
                self.salience_gate,
                llm_client=self
            )
        return self._consolidation_service
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs):
        _ = self.curation_service
        self.last_trace = None

        # 1. Inject Task Context
        triggered_context = self.search_service.get_triggered_tasks_context(self.user_id)
        if triggered_context:
            messages.insert(0, {"role": "system", "content": triggered_context})

        # 2. Inject Explicit Tool-Use Instructions
        tool_system_msg = {
            "role": "system", 
            "content": (
                "You are a helpful assistant with access to a long-term memory tool called 'search_memory'.\n\n"
                "CRITICAL RULES:\n"
                "1. If the user asks about their history, preferences, or specific facts you don't know, YOU MUST USE 'search_memory'.\n"
                "2. DO NOT ASK FOR PERMISSION to search. Just use the tool immediately.\n"
                "3. Do NOT say 'I don't have access to your personal information'. Use the tool instead.\n\n"
                "SEARCH TIERS:\n"
                "- 'fast': Simple fact lookups (e.g., 'What is my name?').\n"
                "- 'balanced': Standard questions (e.g., 'What did I say about X?').\n"
                "- 'deep': Complex reasoning.\n"
            )
        }
        
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] += f"\n\n{tool_system_msg['content']}"
        else:
            messages.insert(0, tool_system_msg)

        # 3. Consolidation Trigger
        user_query = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""
        if user_query:
            consolidated_text = user_query
            consolidated_text = re.sub(r'\bMy\s+', 'The user\'s ', consolidated_text, flags=re.IGNORECASE)
            consolidated_text = re.sub(r'\bI\'m\s+', 'The user is ', consolidated_text, flags=re.IGNORECASE)
            self.consolidation_service.consolidate(consolidated_text, self.user_id)

        completion_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "tools": self.tool_schema,
            "tool_choice": "auto",
            "stream": stream
        }
        completion_kwargs.update(kwargs)
        
        if stream:
            return self._stream_chat(completion_kwargs, user_query)

        try:
            response = self.client.chat.completions.create(**completion_kwargs)
            response_message = response.choices[0].message
        except Exception as e:
            print(f"[LM Studio] Connection error: {e}")
            return "Error: Could not connect to LM Studio."

        if not response_message.tool_calls:
            final_response = response_message.content
        else:
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except:
                    function_args = {}

                content_result = ""
                if function_name == "search_memory":
                    try:
                        res = self.search_service.search(
                            query=function_args.get("query", ""), 
                            user_id=self.user_id, 
                            search_tier=function_args.get("search_tier", "balanced"),
                            llm_client=self
                        )
                        content_result = res["result"]
                        self.last_trace = res["trace"]
                    except Exception as e:
                        content_result = f"Error: {e}"
                elif function_name == "schedule_task":
                    try:
                        import dateutil.parser
                        ts = dateutil.parser.parse(function_args.get("due_date")).timestamp()
                        tid = self.graph_storage.add_task(function_args.get("task_description"), ts, self.user_id)
                        content_result = f"Task scheduled. ID: {tid}"
                    except Exception as e:
                        content_result = f"Error: {e}"

                messages.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": content_result
                })

            try:
                clean_kwargs = {k: v for k, v in completion_kwargs.items() if k not in ['tools', 'tool_choice']}
                clean_kwargs['messages'] = messages
                resp2 = self.client.chat.completions.create(**clean_kwargs)
                final_response = resp2.choices[0].message.content
            except Exception as e:
                final_response = "Error processing tool results."

        return final_response

    def _stream_chat(self, completion_kwargs: dict, user_query: str):
        try:
            stream = self.client.chat.completions.create(**completion_kwargs)
            full_response = ""
            tool_calls_buffer = []
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    yield delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index is not None:
                            while len(tool_calls_buffer) <= tc.index:
                                tool_calls_buffer.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                            cur = tool_calls_buffer[tc.index]
                            if tc.id: cur["id"] = tc.id
                            if tc.function.name: cur["function"]["name"] = tc.function.name
                            if tc.function.arguments: cur["function"]["arguments"] += tc.function.arguments
        except Exception as e:
            yield f"Error streaming: {e}"

    def analyze_and_extract_knowledge(self, text: str) -> Dict:
        """
        Robust extraction using Regex to find JSON in local model output.
        """
        from datetime import datetime
        
        system_prompt = f"""
        /no_think

You are a Knowledge Graph Extractor.
Analyze the text and output a JSON object with these keys:
1. "facts": list of objects {{ "fact": "string", "importance_score": float (0.0-1.0), "expiration_date": "ISO8601 string" or null }}
2. "entities": list of objects {{ "name": "string", "type": "string" }}
3. "relationships": list of objects {{ "subject": "string", "predicate": "string", "object": "string" }}

Output ONLY valid JSON. Do not use <think> tags.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content
            
            # Robust JSON Extraction via Regex
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                cleaned_content = json_match.group(0)
                return json.loads(cleaned_content)
            else:
                # Fallback if no brackets found
                return {"facts": [], "entities": [], "relationships": []}
            
        except Exception as e:
            print(f"[LM Studio] Extraction error: {e}")
            return {"facts": [{"fact": text}], "entities": [], "relationships": []}

    def extract_query_entities(self, query: str) -> List[str]:
        system_prompt = 'Identify key entities (nouns) in the query. Return JSON: {"entities": ["name1", "name2"]}'
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get("entities", [])
            return []
        except:
            return []

    def update_from_text(self, text_block: str):
        self.consolidation_service.consolidate(text_block, self.user_id)

    def synthesize_answer(self, question: str, return_object: bool = False):
        search_output = self.search_service.search(query=question, user_id=self.user_id, search_tier="deep", llm_client=self)
        context = search_output["result"]
        self.last_trace = search_output["trace"]
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        answer = resp.choices[0].message.content
        if return_object:
            from ..observability import AnswerObject
            return AnswerObject(question=question, answer=answer, context=context, trace=self.last_trace)
        return answer

    def close(self):
        try:
            if self._curation_service: self._curation_service.stop()
            if self._vector_storage: self._vector_storage.close()
            if self._graph_storage: self._graph_storage.close()
        except: pass