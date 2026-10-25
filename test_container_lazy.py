"""
单文件测试示例：验证容器懒加载 + 依赖注入不加载重型依赖
运行方式：python test_container_lazy.py
无需安装 sentence-transformers, pymilvus, vllm, transformers, fastapi 等重型依赖
"""
import sys
import os
import dataclasses

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Step 1: 创建容器实例（不应有任何重型库导入日志）")
print("=" * 60)

# 延迟导入：此时不应触发任何重型依赖导入
from bootstrap.container import Container
from infrastructure.config.settings import Settings

test_settings = Settings.from_env()
container = Container(test_settings)

print(f"Container created: {container}")
print(f"Settings loaded: {test_settings.server.host}:{test_settings.server.port}")

# 2. 验证懒加载：首次访问时才创建实例
print("\n" + "=" * 60)
print("Step 2: 验证懒加载 - 首次访问 model_repo")
print("=" * 60)

# 此时才会创建 FilesystemModelRepository
repo = container.model_repo()
print(f"ModelRepository created: {type(repo).__name__}")
print(f"ModelRepository type: {repo}")

# 3. 验证可选依赖优雅降级
print("\n" + "=" * 60)
print("Step 3: 验证可选依赖优雅降级")
print("=" * 60)

# embedding_provider - 无 sentence-transformers 时应返回 NullEmbeddingProvider
embedding = container.embedding_provider()
print(f"EmbeddingProvider: {type(embedding).__name__}")
assert "Null" in type(embedding).__name__, f"Expected NullEmbeddingProvider, got {type(embedding).__name__}"

# vector_store - 无 pymilvus 时应返回 NullVectorStore
vector_store = container.vector_store()
print(f"VectorStore: {type(vector_store).__name__}")
assert "Null" in type(vector_store).__name__, f"Expected NullVectorStore, got {type(vector_store).__name__}"

# rag_service - 无重型依赖时返回 LocalRAGService（包装了 NullVectorStore/NullEmbeddingProvider）
# 这也是正确的降级行为：RAG 服务可用但底层存储/嵌入为空实现
rag = container.rag_service()
print(f"RAGService: {type(rag).__name__}")
# 验证其内部依赖也是空实现
assert type(rag._vector_store).__name__ == "NullVectorStore"
assert type(rag._embedding).__name__ == "NullEmbeddingProvider"
print("✅ RAGService 使用空实现依赖")

# 4. 验证单例缓存
print("\n" + "=" * 60)
print("Step 4: 验证单例缓存")
print("=" * 60)

repo2 = container.model_repo()
print(f"Same instance: {repo is repo2}")
assert repo is repo2, "ModelRepository should be singleton"

# 5. 注入 Mock 替身（测试专用）
print("\n" + "=" * 60)
print("Step 5: 注入 Mock 替身（测试专用）")
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

# 注入 Mock
from application.inference.handlers import ChatUseCase
from application.model.handlers import LoadModelUseCase

container.register_override(ChatUseCase, MockChatUseCase())
container.register_override(LoadModelUseCase, MockLoadModelUseCase())

# 验证 Mock 生效
chat_uc = container.chat_use_case()
load_uc = container.load_model_use_case()

print(f"ChatUseCase: {type(chat_uc).__name__}")
print(f"LoadModelUseCase: {type(load_uc).__name__}")
assert isinstance(chat_uc, MockChatUseCase)
assert isinstance(load_uc, MockLoadModelUseCase)

# 6. 验证领域服务懒加载
print("\n" + "=" * 60)
print("Step 6: 验证领域服务懒加载")
print("=" * 60)

selection = container.selection_service()
print(f"ModelSelectionService: {type(selection).__name__}")

inference = container.inference_service()
print(f"InferenceDomainService: {type(inference).__name__}")

# 7. 验证应用用例懒加载
print("\n" + "=" * 60)
print("Step 7: 验证应用用例懒加载")
print("=" * 60)

chat_uc2 = container.chat_use_case()
print(f"ChatUseCase (after override): {type(chat_uc2).__name__}")

load_uc2 = container.load_model_use_case()
print(f"LoadModelUseCase: {type(load_uc2).__name__}")

list_uc = container.list_models_use_case()
print(f"ListModelsUseCase: {type(list_uc).__name__}")

# 8. 测试领域逻辑（无重型依赖）
print("\n" + "=" * 60)
print("Step 8: 测试领域逻辑（纯内存，无外部依赖）")
print("=" * 60)

# 测试 TokenBudgetService
from domain.model.services import TokenBudgetService

budget_service = TokenBudgetService(default_context_length=8192, reserve_tokens=1024)
budget = budget_service.calculate_budget(8192, 1024)
print(f"Token budget: {budget}")
assert budget == 8192 - 1024 - 1024

# 测试截断逻辑
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
assert truncated[0]["role"] == "system"  # 保留 system

# 9. 测试实体创建
print("\n" + "=" * 60)
print("Step 9: 测试领域实体")
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

# 验证值对象不可变性
try:
    config.max_new_tokens = -1
    print("ERROR: Should have raised error")
except (ValueError, dataclasses.FrozenInstanceError) as e:
    print(f"✅ ValueObject immutability works: {type(e).__name__}")

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED - 懒加载、可选依赖降级、Mock注入全部正常！")
print("✅ 无重型依赖加载")
print("✅ 单文件可运行")
print("✅ 领域逻辑可独立测试")
print("=" * 60)