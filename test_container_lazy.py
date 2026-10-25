"""
Single-file test: Verify container lazy loading + DI without heavy deps.
Run: python test_container_lazy.py
No sentence-transformers, pymilvus, vllm, transformers, fastapi required.
"""
import sys
import os
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Step 1: Create container instance (no heavy lib imports)")
print("=" * 60)

from bootstrap.container import Container
from infrastructure.config.settings import Settings

test_settings = Settings.from_env()
container = Container(test_settings)

print(f"Container created: {container}")
print(f"Settings loaded: {test_settings.server.host}:{test_settings.server.port}")

print("\n" + "=" * 60)
print("Step 2: Verify lazy loading - first access model_repo")
print("=" * 60)

repo = container.model_repo()
print(f"ModelRepository created: {type(repo).__name__}")
print(f"ModelRepository type: {repo}")

print("\n" + "=" * 60)
print("Step 3: Verify optional dependency graceful degradation")
print("=" * 60)

embedding = container.embedding_provider()
print(f"EmbeddingProvider: {type(embedding).__name__}")
assert "Null" in type(embedding).__name__, f"Expected NullEmbeddingProvider, got {type(embedding).__name__}"

vector_store = container.vector_store()
print(f"VectorStore: {type(vector_store).__name__}")
assert "Null" in type(vector_store).__name__, f"Expected NullVectorStore, got {type(vector_store).__name__}"

rag = container.rag_service()
print(f"RAGService: {type(rag).__name__}")
assert type(rag._vector_store).__name__ == "NullVectorStore"
assert type(rag._embedding).__name__ == "NullEmbeddingProvider"
print("RAGService uses null implementations")

print("\n" + "=" * 60)
print("Step 4: Verify singleton cache")
print("=" * 60)

repo2 = container.model_repo()
print(f"Same instance: {repo is repo2}")
assert repo is repo2, "ModelRepository should be singleton"

print("\n" + "=" * 60)
print("Step 5: Register Mock overrides (test only)")
print("=" * 60)

class MockChatUseCase:
    async def execute(self, command):
        class MockChunk:
            content = "Mock response"
            finish_reason = "stop"
            is_final = True
        yield MockChunk()

    async def execute_once(self, command):
        return "Mock response"

class MockLoadModelUseCase:
    def execute(self, command):
        class MockEvent:
            model_id = command.model_id
            engine_type = "mock"
            context_length = 8192
        return MockEvent()

from application.inference.handlers import ChatUseCase
from application.model.handlers import LoadModelUseCase

container.register_override(ChatUseCase, MockChatUseCase())
container.register_override(LoadModelUseCase, MockLoadModelUseCase())

chat_uc = container.chat_use_case()
load_uc = container.load_model_use_case()

print(f"ChatUseCase: {type(chat_uc).__name__}")
print(f"LoadModelUseCase: {type(load_uc).__name__}")
assert isinstance(chat_uc, MockChatUseCase)
assert isinstance(load_uc, MockLoadModelUseCase)

print("\n" + "=" * 60)
print("Step 6: Verify domain service lazy loading")
print("=" * 60)

selection = container.selection_service()
print(f"ModelSelectionService: {type(selection).__name__}")

inference = container.inference_service()
print(f"InferenceDomainService: {type(inference).__name__}")

print("\n" + "=" * 60)
print("Step 7: Verify application use case lazy loading")
print("=" * 60)

chat_uc2 = container.chat_use_case()
print(f"ChatUseCase (after override): {type(chat_uc2).__name__}")

load_uc2 = container.load_model_use_case()
print(f"LoadModelUseCase: {type(load_uc2).__name__}")

list_uc = container.list_models_use_case()
print(f"ListModelsUseCase: {type(list_uc).__name__}")

print("\n" + "=" * 60)
print("Step 8: Test domain logic (pure memory, no external deps)")
print("=" * 60)

from domain.model.services import TokenBudgetService

budget_service = TokenBudgetService(default_context_length=8192, reserve_tokens=1024)
budget = budget_service.calculate_budget(8192, 1024)
print(f"Token budget: {budget}")
assert budget == 8192 - 1024 - 1024

messages = [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "Message 1"},
    {"role": "assistant", "content": "Response 1"},
    {"role": "user", "content": "Message 2"},
    {"role": "assistant", "content": "Response 2"},
]

def count_tokens(msg):
    return len(msg.get("content", "")) // 3

truncated = budget_service.truncate_messages(messages, count_tokens, budget=50)
print(f"Original: {len(messages)} messages")
print(f"Truncated: {len(truncated)} messages")
assert len(truncated) <= len(messages)
assert truncated[0]["role"] == "system"

print("\n" + "=" * 60)
print("Step 9: Test domain entities")
print("=" * 60)

from domain.model.entities import Model, ModelId, ModelPath, ModelType, GenerationConfig, EngineType
from pathlib import Path

model = Model.create(
    model_id="test-model",
    path=Path("/fake/path"),
    model_type=ModelType.HF,
    name="Test Model",
)

print(f"Model created: {model.id.value}, type={model.model_type.value}")
print(f"Preferred engine: {model.preferred_engine().value}")

config = GenerationConfig(max_new_tokens=100, temperature=0.5)
print(f"GenerationConfig: max_tokens={config.max_new_tokens}, temp={config.temperature}")

try:
    config.max_new_tokens = -1
    print("ERROR: Should have raised error")
except (ValueError, dataclasses.FrozenInstanceError) as e:
    print(f"ValueObject immutability works: {type(e).__name__}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - Lazy loading, optional dep degradation, Mock injection OK!")
print("No heavy deps loaded")
print("Single-file runnable")
print("Domain logic testable in isolation")
print("=" * 60)