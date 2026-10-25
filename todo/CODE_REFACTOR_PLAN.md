# 代码规范整改计划

基于 `CODE_PRINCIPLES` 规范对项目进行全面整改。

## 核心原则违规清单

### 1. 全局状态与单例模式滥用（严重）
| 文件 | 问题 | 规范条款 |
|------|------|----------|
| `model_providers/model_provider.py:41` | `local_model = None` 全局单例持有业务状态 | 单例不得保存业务状态、禁止隐式共享状态 |
| `model_providers/model_provider.py:27` | `models = discover_models()` 模块级执行，隐式依赖文件系统 | 禁止隐式共享状态、配置必须可注入 |
| `controllers/llm_controller.py:14` | `import globals as backend_globals` 全局状态容器 | 禁止隐式共享状态 |

### 2. 硬编码配置与环境变量直接访问（严重）
| 文件 | 问题 | 规范条款 |
|------|------|----------|
| `model_providers/model_provider.py:70` | 硬编码默认模型名 `"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"` | 禁止硬编码环境变量、路径、魔法数字 |
| `model_providers/model_provider.py:260-264` | 直接 `os.getenv("EMBEDDING_DEVICE")`、`os.getenv("CACHE_PATH")` | 配置必须可注入、可替换、可测试 |
| `model_providers/unified_model_loader.py` | 多处直接读取环境变量 | 同上 |
| `constants.py` | 100+ 个 `os.getenv()` 调用，配置散落 | 配置必须集中管理、可注入 |

### 3. 业务层直接依赖第三方库/基础设施（严重）
| 文件 | 直接依赖 | 规范条款 |
|------|---------|----------|
| `model_providers/model_provider.py` | `sentence_transformers`, `torch`, `transformers` | 业务层禁止依赖第三方库、框架或基础设施实现 |
| `model_providers/transformers_engine.py` | `transformers`, `torch` | 同上 |
| `model_providers/llama_cpp_engine.py` | `httpx`, `subprocess`, `docker` | 同上 |
| `model_providers/vllm_engine.py` | `vllm`, `transformers` | 同上 |
| `utils/utils.py` | `torch`, `psutil`, `PyPDF2` | 同上 |

### 4. 文件/函数过大（中等）
| 文件 | 行数 | 规范 | 状态 |
|------|------|------|------|
| `utils/utils.py` | 449 | ≤500 | 接近超标 |
| `model_providers/multimodal_provider.py` | 437 | ≤500 | 接近超标 |
| `model_providers/model_provider.py` | 380 | ≤500 | 需拆分 |
| `model_providers/unified_model_loader.py` | 344 | ≤500 | 需拆分 |
| `model_providers/llama_cpp_engine.py` | 317 | ≤500 | 需拆分 |
| `controllers/llm_controller.py` | 284 | ≤500 | 需拆分 |
| `model_providers/vllm_engine.py` | 254 | ≤500 | 可接受但需优化 |

**函数超标示例**：
- `load_model` (40行) → 拆分为配置构建、加载执行、引擎创建
- `format_messages` (30行) → 拆分为类型归一化、内容提取
- `smart_truncate_messages` (30行) → 拆分为 token 计算、截断策略
- `generate_next_token` (25行) → 拆分为状态管理、异常处理

### 5. 依赖具体实现而非接口（严重）
| 现状 | 要求 |
|------|------|
| 直接 `new UnifiedModelLoader()` | 依赖 `ModelLoader` 接口 |
| 直接 `new Scheduler()` | 依赖 `TaskScheduler` 接口 |
| 直接 `new SentenceTransformer()` | 依赖 `EmbeddingProvider` 接口 |
| 直接 `new LocalRAG()` | 依赖 `RAGService` 接口 |

### 6. 异常处理不规范（中等）
| 文件 | 问题 | 规范 |
|------|------|------|
| `model_provider.py:316-323` | 吞异常，仅记录日志返回 None | 禁止吞异常、异常必须可分类、可定位、可追责 |
| `unified_model_loader.py` | 多处 `except Exception` 空捕获 | 同上 |
| `llama_cpp_engine.py` | 多处裸 `except:` | 同上 |

### 7. 命令查询不分离（中等）
| 方法 | 问题 |
|------|------|
| `load_model()` | 既初始化引擎(副作用)又隐式返回 engine |
| `chat()` / `complete()` | 既格式化消息又执行生成 |
| `format_prompt()` | 既格式化又依赖 engine.tokenizer 副作用 |

### 8. 过度注释（轻微）
多处「解释性注释」应通过命名自解释，如：
- `get_available_memory()` 中的中文注释
- `discover_models()` 中的逻辑注释
- 控制器中的流程注释

### 9. DDD 分层架构缺失（架构级）
| 现状 | 目标分层 |
|------|---------|
| 控制器直接依赖模型提供者、RAG、第三方库 | **接口层** → **应用层** → **领域层** → **基础设施层** |
| 模型加载、推理、调度混在一起 | 领域服务、应用服务、基础设施适配器分离 |
| 配置散落在 constants、环境变量、硬编码 | 统一配置对象，依赖注入 |

---

## 整改策略：分层架构重构

### 目标架构

```
local-llm-provider/
├── domain/                    # 领域层（纯业务逻辑，零外部依赖）
│   ├── model/
│   │   ├── entities.py        # Model, ModelConfig, GenerationConfig
│   │   ├── value_objects.py   # ModelId, ModelPath, TokenCount
│   │   ├── repositories.py    # ModelRepository 接口
│   │   └── services.py        # ModelDomainService 领域服务
│   ├── inference/
│   │   ├── entities.py        # InferenceRequest, InferenceResponse
│   │   ├── ports.py           # InferenceEngine 接口（已存在）
│   │   └── services.py        # InferenceDomainService
│   ├── embedding/
│   │   ├── ports.py           # EmbeddingProvider 接口
│   │   └── services.py
│   ├── rag/
│   │   ├── entities.py
│   │   ├── ports.py           # RAGRepository, VectorStore 接口
│   │   └── services.py
│   └── scheduling/
│       ├── ports.py           # TaskScheduler 接口
│       └── services.py
│
├── application/               # 应用层（编排领域服务，无业务逻辑）
│   ├── model/
│   │   ├── commands.py        # LoadModelCommand, UnloadModelCommand
│   │   ├── queries.py         # GetAvailableModelsQuery
│   │   └── handlers.py        # 命令/查询处理器
│   ├── inference/
│   │   ├── commands.py        # GenerateCommand
│   │   └── handlers.py
│   └── rag/
│       └── handlers.py
│
├── infrastructure/            # 基础设施层（第三方库适配实现）
│   ├── model/
│   │   ├── loader/
│   │   │   ├── transformers_loader.py
│   │   │   ├── llama_cpp_loader.py
│   │   │   ├── vllm_loader.py
│   │   │   └── factory.py     # ModelLoaderFactory
│   │   ├── repository/
│   │   │   └── filesystem_model_repository.py
│   │   └── config/
│   │       └── model_config_provider.py
│   ├── inference/
│   │   ├── transformers_engine.py  # 实现 InferenceEngine
│   │   ├── llama_cpp_engine.py
│   │   ├── vllm_engine.py
│   │   └── engine_factory.py
│   ├── embedding/
│   │   └── sentence_transformer_provider.py  # 实现 EmbeddingProvider
│   ├── rag/
│   │   ├── milvus_vector_store.py
│   │   ├── elasticsearch_store.py
│   │   └── local_rag_service.py
│   ├── scheduling/
│   │   └── async_scheduler.py
│   └── config/
│       └── settings.py        # 统一配置管理
│
├── interfaces/                # 接口层（HTTP/gRPC/CLI 适配）
│   ├── http/
│   │   ├── routes/
│   │   │   ├── chat_routes.py
│   │   │   ├── model_routes.py
│   │   │   └── rag_routes.py
│   │   ├── dto/
│   │   │   ├── chat_dto.py
│   │   │   └── model_dto.py
│   │   └── main.py
│   └── cli/
│       └── commands.py
│
├── bootstrap/                 # 启动配置（依赖注入容器）
│   ├── container.py
│   └── wiring.py
│
└── tests/
    ├── domain/
    ├── application/
    └── infrastructure/
```

---

## 实施阶段

### Phase 1: 基础设施层接口化（优先级：P0）
- [ ] 定义领域接口：`ModelRepository`, `EmbeddingProvider`, `TaskScheduler`, `RAGService`
- [ ] 创建统一配置对象 `Settings` 替代 `constants.py` + 环境变量
- [ ] 将 `UnifiedModelLoader` 拆分为 `ModelLoader` 接口 + 多实现 + 工厂
- [ ] 将 `Scheduler` 抽象为 `TaskScheduler` 接口
- [ ] 将 `SentenceTransformer` 封装为 `EmbeddingProvider` 接口实现

### Phase 2: 移除全局状态（优先级：P0）
- [ ] 移除 `model_provider.py` 中的 `local_model` 全局单例
- [ ] 移除 `models = discover_models()` 模块级执行
- [ ] 移除 `globals.py` 全局状态容器
- [ ] 引入依赖注入容器管理生命周期

### Phase 3: 领域层建模（优先级：P1）
- [ ] 定义实体：`Model`, `InferenceSession`, `GenerationConfig`
- [ ] 定义值对象：`ModelId`, `ModelPath`, `TokenCount`, `ModelType`
- [ ] 定义领域服务：`ModelSelectionService`, `TokenBudgetService`
- [ ] 定义领域事件：`ModelLoaded`, `ModelUnloaded`, `GenerationCompleted`

### Phase 4: 应用层编排（优先级：P1）
- [ ] 创建命令/查询处理器
- [ ] 实现用例：`LoadModelUseCase`, `GenerateUseCase`, `ChatUseCase`
- [ ] 移除控制器中的业务逻辑，仅做 DTO 转换和路由

### Phase 5: 接口层清理（优先级：P1）
- [ ] 拆分 `llm_controller.py` 为多个路由文件
- [ ] 定义稳定的 DTO 结构
- [ ] 移除控制器中的直接依赖，改为依赖应用层 UseCase

### Phase 6: 代码质量提升（优先级：P2）
- [ ] 拆分超长函数（>20行）
- [ ] 拆分超大文件（>500行）
- [ ] 移除解释性注释，通过命名自解释
- [ ] 统一异常体系，定义领域异常
- [ ] 参数默认不可变，显式校验输入
- [ ] 添加单元测试覆盖核心领域逻辑

---

## 约束与原则

1. **增量重构**：每步保持测试通过，不大爆破重写
2. **接口优先**：先定接口，再实现，再替换调用方
3. **配置外部化**：所有配置通过 `Settings` 注入，禁止直接 `os.getenv`
4. **依赖倒置**：业务层依赖抽象，基础设施层实现抽象
5. **单一职责**：每个类/函数只做一件事
6. **不可变性**：参数默认不可变，状态变更显式建模

---

## 验收标准

| 指标 | 当前 | 目标 |
|------|------|------|
| 文件最大行数 | 449 | ≤ 300 |
| 函数最大行数 | ~40 | ≤ 20 |
| 全局可变状态 | 5+ | 0 |
| 硬编码配置项 | 20+ | 0 |
| 业务层第三方依赖 | 多处 | 0 |
| 接口抽象覆盖率 | ~10% | 100% 核心能力 |
| 单元测试覆盖核心域 | 无 | ≥ 80% |