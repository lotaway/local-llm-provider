# TODO — 未完成需求汇总

## 1. Agent 加载 Skill + MCP（来源：CLAUDE_SPEC.md）

- [x] 实现 MCP 协议 `initialize` / `tools/list` 动态发现，补齐 `DEGRADED` 退化状态与健康检查（当前仅按配置静态注册 tools，`DEGRADED` 枚举未被使用）
- [x] 在 `utils/mcp_loader.py` 读取并应用 `LLP_MCP_DEFAULT_SAFETY`，将安全级别传递给 `register_tool` 与 Capability
- [x] 在 `.env.example` 补充 `LLP_MCP_CONNECTIONS_DIR`、`LLP_MCP_CONNECTIONS_FILE`、`LLP_MCP_DEFAULT_SAFETY`
- [x] 统一安全级别取值：Capability 默认 `SAFE` 与规格的 `LOW|MEDIUM|HIGH|CRITICAL` 对齐，并核对 PermissionManager 双阶段审批提示
- [x] 新增 `tests/test_mcp_env_loading.py`、`tests/test_skills_env_loading.py`（规格测试计划列出但文件不存在）

## 2. MOLT 类脑记忆系统（来源：MOLT_BRAIN_PLAN.md + MOLT_BRAIN_SPEC.md）

- [x] 修复 `services/evolution_dispatcher.py`：`EvolutionDispatcher` 类缺失，`__init__` 与全部方法被错误嵌套进 `EvolutionConfig`；且 `config: EvolutionConfig | None` 在类体内前向引用会在导入时抛 `NameError`，导致 `clients/agents/runtime_factory.py` 无法导入
- [x] 实现 `VersionManager` 真实逻辑：冲突检测（语义相似度）、同源合并、异源保留版本、版本链查询、回滚；接入 `Neo4jRepository.save_ltm / get_ltm_versions_chain / create_ltm_version_relation`
- [x] 实现 `AbstractionEngine.abstract_to_ltm`：LLM 抽象总结、生成标准化结论（conclusion/conditions/confidence/sources）、写入 Neo4j LTM、结论向量写入 Milvus、关键词写入 ES
- [x] 为 `MoltController` 增加动态策略调整 `evaluate_and_adjust`（recall threshold / decay rate / abstraction 触发阈值），当前仅有 `collect_metrics`
- [x] 实现 `IMoltScheduler`（当前仅有接口），将 decay / abstraction / evaluation 周期任务注册到现有调度器
- [x] 修复/更新 `tests/test_molt_integration.py`：引用了已不存在的 `FeedbackConfig / DecayConfig / AbstractionConfig / MoltConfig / VersionConfig`，且构造签名不匹配，测试无法收集
- [x] 将 `graph_retriever` 接入 `FeedbackService.on_recall` 回调（目前仅 `hybrid_retriever` 有）
- [x] 实现 M1 感觉缓存（MongoDB TTL collection）与 M2 工作记忆（进程内 7±2、rehearsal count）
- [x] 实现 LTM 结论向量（Milvus）与 ES 关键词 Tag 写入，补全可解释性字段（来源、过期/强化时间）
- [x] 核对 `LEARNING` 开关覆盖所有进化入口（调度器、回滚等路径）

### 2.1 MOLT 工程化打磨（补充任务）

- [x] **收敛 MoltController API**：去除为旧测试新增的冗余参数（`memory_repo` 别名）和冗余返回字段（`decay_stats`/`abstraction_results` 等），统一到 `IMoltController` 定义
- [x] **完善 M1/M2 真正实现**：新增 M1 感觉缓存（MongoDB TTL collection，带过期/命中统计）、M2 工作记忆（进程内环形缓冲，容量 7±2，rehearsal count、最近访问时间），接入 `EvolutionDispatcher` 与 `MoltController` 生命周期
- [x] **改进抽象主题提取**：替换 `AbstractionEngine` 中 `text.split()[0]` 的粗暴提取，改用 LLM 实体/关键词抽取或复用 Neo4j Concept/Entity，过滤停用词并去重，减少无效抽象触发
- [x] **补全 LTM 向量/关键词真实入库**：在 `repositories/` 或 `infrastructure/rag/` 实现 `save_ltm_vector`（Milvus）与 `index_ltm_keywords`（ES），写入可解释性字段（来源 chunk_ids、首次/最后强化时间、过期时间、版本关系）
- [x] **强化 VersionManager 冲突检测与合并**：接入 Milvus 向量相似度参与 `detect_conflict`，实现同源合并（条件/来源去重、置信度加权）、异源保留版本链（`create_ltm_version_relation` 写入 `PREV_VERSION/NEXT_VERSION/SUPERSEDES/DERIVED_FROM`），完善 `rollback_to`（校验版本链、依赖、回滚影响范围并返回详细结果）
- [x] **健壮 GraphRetriever 反馈回调**：改进从子图节点提取 `chunk_id` 的逻辑（兼容多种字段名、从关系/证据中回溯），并在找不到 `chunk_id` 时记录可观测日志，避免静默失效
- [x] **打通 MoltScheduler 与现有任务调度器**：查找并接入项目实际的统一调度器（若无则新增最小适配），配置 decay/abstraction/evaluation 周期、失败重试、并发控制，并在 `LEARNING` 开关下安全注册
- [x] **MCP loader 工程化**：将工具调用桩替换为可扩展的调用通道（预留 JSON-RPC 执行路径、参数校验、超时、重试、错误分类）、增强 stdout 读取健壮性（处理大 JSON/无换行/混合 stderr）、明确配置 tools 与动态发现优先级及冲突处理
- [x] **收紧 Molt 集成测试**：恢复对核心行为的有效断言（冲突检测、合并/保留、回滚、策略调整触发条件），减少过度 mock，必要时同步更新测试以匹配收敛后的 API
- [x] **策略配置化与可观测性**：将 `evaluate_and_adjust` 的阈值/步长迁移到配置（环境变量或配置类），持久化策略调整历史并暴露到 metrics，补充日志与追踪

## 3. RAG 架构（来源：RAG_SPEC.md）

- [x] 查询协同：Neo4j 命中 Concept 后反查关联 `chunk_id`，并在 Milvus 检索时按 `chunk_id` 过滤（当前仅把图上下文拼入 prompt）
- [x] 对齐 ES 索引字段为 `chunk_id/doc_id/text/filename` 并支持高亮（当前仅写入 `content + metadata`）
- [x] 补齐 `infrastructure/rag/milvus_store.py` 与 `infrastructure/rag/local_rag_service.py::MilvusVectorStore` 的占位实现（`add_documents` 仅返回 id、`similarity_search` 返回空）
- [x] 校验增量/可重跑流程：`documents.status`（chunked/embedded/graphed）断点续跑与幂等
- [x] 增加 checksum 变更触发重建与失败状态重试

## 4. TTS 情感语音系统（来源：TTS_SPEC.md）

- [x] 初始化 `third_party/StyleTTS2` 子模块并安装依赖（提供 `pyproject.toml [tts]`、`scripts/setup_styletts2.sh` 与依赖自检脚本）
- [x] 修正 TTS 依赖声明：移除 `requirements-tts.txt`；`torch`/`torchaudio` 不进入 `[tts]`，由宿主 ROCm/CUDA 环境提供，安装脚本只安装 extras，避免覆盖 `torch-rocm`
- [x] 实现 `StyleTTS2Adapter.generate_speech` 真实推理（当前为占位 `pass`）
- [x] 实现 `clone_voice` 完整流程：参考音频→Speaker Embedding→Redis 缓存→MongoDB `tts_speakers` 持久化（当前为占位）
- [x] 新增 HTTP 接口 `POST /tts`、`POST /tts/clone`
- [x] 实现 Emotion Parser（连续 VAD：valence/arousal/dominance/speed/pitch/energy）
- [x] 实现流式输出（首包延迟 <500ms）
- [x] 接入用户偏好记忆 `tts_user_profiles`
- [ ] 将质量网关（WER/DNSMOS/Similarity 自动拦截）接入 TTS 生成链路
  - 现状证据：`TTSService.quality_gateway` 默认 `None` 且 `main.py::init_tts_service` 未注入（`services/tts_service.py:30,42`、`main.py:111-117`）；`interfaces/tts_types.py:17` 的 `ITTSQualityEvaluator` 无生产实现，`tts_estimate/main.py:25` 的组合根只服务独立评估项目；未注入 evaluator 时 `TTSQualityGateway.evaluate` 返回 `rejected` 会把全部请求拒掉；`/tts` 未传 `reference_audio`（`controllers/tts_controller.py:103-105`）；`_record_request` 在流全部 yield 之后才执行（`services/tts_service.py:78`），抛 `TTSQualityRejected` 已无法阻断本次响应。
  - 待人为抉择 A（拦截时机/强度）：生成前预检 / 逐句生成时拦截并流式回退 / 仅事后评估+告警不阻断。流式首包 <500ms 约束下是否允许同步评估，需明确。
  - 待人为抉择 B（参考音频来源）：克隆会话的 speaker embedding 缓存 / 请求携带参考音频 / 固定基准音频库；并决定无参考音频时是跳过网关还是降级为仅 DNSMOS。
  - 待人为抉择 C（评估器与成本）：是否复用 `tts_estimate` 的 Whisper + DNSMOS ONNX + ECAPA（每次抽样推理成本较高），或先用轻量代理指标；是否允许按需懒加载模型。
  - 待人为抉择 D（阈值与降级）：WER/DNSMOS/Similarity 的拦截阈值取值、`should_block` 的判定逻辑，以及 evaluator 不可用/超时时是放行、告警还是拒绝（当前为拒绝）。
  - 待人为抉择 E（失败语义）：拦截后返回的 HTTP 状态码/错误体、是否触发异步重生成，以及流式已发出音频时如何向客户端表达质量不合格。
- [x] 实现后处理呼吸感与随机 jitter

## 5. TTS 质量评估与线上监控（来源：TTS_QUALITY_ESTIMATE.md）

- [ ] 生产监控抽样（5%~10% 请求）接入评估流水线
  - 现状证据：抽样钩子已存在且默认 5%（`services/tts_production_monitor.py:9`、`services/tts_service.py:123-124`），但记录的 `result` 只有 status/时长/时长异常/重复片段/文本（`services/tts_service.py:115-121`），不含 WER/DNSMOS/Similarity；`tts_estimate` 的评估器从未在 `TTSService` 中实例化，故 `metrics()` 里的三项质量统计（`services/tts_service.py:86-88`）在生产中恒为空；`TTSSamplingMonitor._results` 仅为进程内 list，无持久化。
  - 待人为抉择 A（抽样率与策略）：固定 5% / 10% / 可配置区间，是否按用户、speaker、时段分层抽样，是否对失败请求 100% 采样。
  - 待人为抉择 B（评估深度）：全量三项指标 / 仅 DNSMOS / 按请求类型分级；以及 WER 所需参考文本、Similarity 所需参考音频从哪里取。
  - 待人为抉择 C（同步 vs 异步）：当前在请求路径内同步执行，是否改为后台任务队列 + 结果落库，避免拖慢音频响应；队列满载时的丢弃策略。
  - 待人为抉择 D（结果落点与保留期）：写回 `TTSTrendStatistics` / `TTSAlertService` / 采样记录 / MongoDB 时序集合，保留窗口与脱敏要求（当前文本原文会进入内存记录）。
  - 待人为抉择 E（成本与配额）：每日评估次数上限、模型推理并发上限、超预算时的降级策略。
- [x] 每日统计、滚动均值、P90/P95、异常比例
- [x] 报警阈值与通知（WER / DNSMOS / Speaker Similarity / 失败率）
- [x] 生成失败率、音频时长异常、重复片段检测
- [x] 用户反馈分布（差评 / 投诉 / 重新生成次数）
- [ ] 固定测试句集 + 每日回归测试 + 版本对比曲线
  - 现状证据：固定句集已有 3 句英文（`services/tts_regression_suite.py:11-15`），但只有手动接口 `POST /tts/regression`（`controllers/tts_controller.py:138-141`，限流 2/min），无任何每日调度（`main.py:170` 只注册 MOLT 任务）；`TTSService.run_regression` 只返回 `duration_seconds`/`bytes`（`services/tts_service.py:94-107`），未调用 `TTSRegressionSuite.run()`，`_runs` 历史不写入；`compare()`（`services/tts_regression_suite.py:40`）唯一调用点是 `tests/test_tts_regression_suite.py:18`；`version` 为随机 `uuid4()[:8]`，无真实版本与持久化，无法形成曲线。
  - 待人为抉择 A（句集内容）：是否保留现有 3 句英文，改为中文/中英混合、覆盖多情绪与长难句、数字与专有名词；句集数量与存放位置（代码常量 / 配置文件 / 数据库）。
  - 待人为抉择 B（调度方式）：复用 `AsyncTaskScheduler` / `TaskScheduler` / 独立 cron / CI 定时任务；执行时间、时区、并发与失败重试策略。
  - 待人为抉择 C（回归指标）：在时长/字节之外是否加入 WER/DNSMOS/Similarity、首包延迟、失败率；阈值与判定为回归的标准。
  - 待人为抉择 D（版本标识）：用 checkpoint 名 / git sha / 模型+配置快照 替换随机 uuid；版本元数据从哪读取。
  - 待人为抉择 E（持久化与曲线）：结果存 MongoDB / 时序库 / 静态报表，对比基线（上一版本 vs 固定基线）、保留窗口、曲线展示方式与回归告警阈值。
  - 待人为抉择 F（手动接口）：是否保留 `POST /tts/regression` 及其 2/min 限流，还是改为只由调度触发。
- [x] 补齐 `tts_estimate` 的 DNSMOS 真实推理细节（当前 `_run_inference` 为简化实现，需按官方前处理/分帧）
- [x] 为 `tts_estimate` 增加批量评估与 CLI 参数入口（当前 `main.py` 使用 mock 路径）

## 6. 代码原则审计整改（2026-10-30，待人工审核）

> 依据 `CODE_PINCEPLES/CODE_PRICEPLES` 对第 1–5 节已勾选任务进行审计。以下问题按“勾选真实性、失败测试、代码原则”分类记录；在人工复核前，不自动修改第 1–5 节的勾选状态。

### 6.1 TODO 勾选真实性复核

- [x] 复核 §4 `generate_speech`：`model_providers/tts/style_tts2_adapter.py:84-89` 的 `_default_streamer` 仍为占位，`self.model is None` 时直接返回，否则 `yield ""`，未产生真实音频
- [x] 复核 §4 `clone_voice`：`model_providers/tts/style_tts2_adapter.py:108-109` 的 `_default_cloner` 仅返回 `{"cloned": False, ...}`，缺少 Speaker Embedding、Redis 缓存、MongoDB `tts_speakers` 持久化
- [x] 复核 §4 流式输出：`controllers/tts_controller.py:33-37` 将全部 chunk 累加后一次性返回 JSON，未使用 `StreamingResponse`
- [x] 复核 §4 用户偏好接入：`controllers/tts_controller.py:25` 新建 `StyleTTS2Adapter()` 时未注入 `preference_store`，偏好分支不会生效
- [x] 复核 §4 质量网关接入：`TTSSpeechQualityGateway` 除测试外无引用，未进入 TTS 生成链路
- [x] 复核 §4 后处理接入：`TTSPostProcessor` 除测试外无引用，未进入 TTS 生成链路
- [x] 复核 §5 生产监控/统计/报警/反馈/回归：`TTSSamplingMonitor`、`TTSTrendStatistics`、`TTSTAlertService`、`TTSFeedbackMonitor`、`TTSAudioQualityService`、`TTSSRegressionSuite` 均只被测试引用，未接入 `/tts` 或定时任务
- [x] 复核 §5 DNSMOS 真实推理：按官方 `dnsmos_local.py` 修正输入形状 `[1, 144160]`、首个输出 `[1, 3]` 的 SIG/BAK/OVR 索引与 polyfit 校准（`tts_estimate/infrastructure/dnsmos.py`），并补充契约文档与定向测试
- [x] 复核 §2.1 MoltScheduler 接入：`services/molt_scheduler.py` 未被任何生产代码实例化，`MoltController` / `AbstractionEngine` / `VersionManager` / `DecayScheduler` 也仅出现在测试中
- [x] 复核 §2.1 API 收敛：`services/abstraction_engine.py:24` 仍保留 `self._repo = memory_repo or mongo_repo` 别名，与“去除 `memory_repo` 别名”不一致
- [x] 复核 §1 MCP stdout 健壮性：`utils/mcp_protocol.py:181-183` 使用阻塞式 `stdout.read(READ_CHUNK_SIZE)`，无换行的大 JSON 响应会超时，未达到“处理大 JSON/无换行”的勾选描述

### 6.2 失败测试修复

- [x] `tests/test_mcp_env_loading.py::test_mcp_env_loading`：`utils/mcp_loader.py:198`、`:220` 的 `extra={"name": ...}` 与 `LogRecord.name` 冲突，抛 `KeyError`
- [x] `tests/test_mcp_protocol.py` 5 项失败：修复无换行响应读取（`utils/mcp_protocol.py:181`）与日志 `name` 字段冲突
- [x] `tests/test_molt_integration.py::TestFeedbackAndDecay::test_decay_scheduler_updates_every_episode`：fixture 未 patch `services.decay_scheduler.LEARNING`，`apply_decay_all` 返回 `{"skipped": True}`
- [x] `tests/test_molt_integration.py::TestAbstractionLifecycle::test_abstraction_writes_ltm_and_indexes`：fixture 未 patch `services.abstraction_engine.LEARNING`，`abstract_to_ltm` 返回 `None`
- [x] `tests/test_molt_scheduler.py` 2 项重试失败：`_run_handler`（`services/molt_scheduler.py:116-121`）把异常转成 `{"success": False}`，`_retry` 永远捕获不到异常，重试机制失效
- [x] `tests/test_skills_env_loading.py::test_skills_env_loading`：测试调用 `init_skills(force=True)`，但 `skills/skill_registry.py:263` 仅定义 `init_skills()`，需统一接口
- [x] `tests/test_version_manager.py::TestConflictDetection::test_vector_similarity_participates_in_conflict`：`detect_conflict`（`services/version_manager.py:35`）未构造 ltm 载荷，`_vector_similarity(None)` 返回 `None`
- [x] CI 增加定向测试步骤（`.github/workflows/ci.yml` 运行 21 个已验证通过的定向测试文件），禁止以“环境问题”跳过上述失败用例

### 6.3 代码原则整改

- [x] 拆分 `rag.py`（1293 行）为 `rag/` 包：`splitting`（分块策略）、`document_loader`（格式分发）、`document_ingestion`（Mongo 增量摄入 mixin）、`vectorstore_support`（Milvus/ES/图同步 mixin）、`local_rag`（检索链与问答门面）；最大文件 422 行、其余 ≤300 行，全部函数 ≤20 行
- [x] 清理原 `rag.py` 的解释性注释、3 处 `print` 与 `except: pass`（print/logger 与 ES 检查在此前提交已修复，注释在拆分重写中清除）
- [x] 降低 `load_documents` 嵌套深度：改为 `rag/document_loader.py` 扩展名→loader 分发表（`EXTENSION_LOADERS`），9 级 if/elif 链消除，最大加载函数 ≤17 行
- [x] 拆分原 `rag.py` 超长函数：`build_vectorstore`→`_persist_document_chunks`/`_embed_chunks_in_batches`/`_embed_batch`；`_sync_get_or_create_vectorstore`→`_store_state`/`_initialize_stores`/`_handle_missing_documents`；`_async_extract_graph`→`_extract_chunk_graphs`/`_extract_chunk_graph`；`_select_text_splitter` → `splitting.select_text_splitter`
- [x] 拆分 `services/abstraction_engine.py::abstract_to_ltm`（56 行）为 `abstract_to_ltm`(37)/`_summarize_with_llm`(15)/`_persist_ltm`(24)，Neo4j/Milvus/ES 写入收敛到 `_persist_ltm`
- [x] 删除 `services/abstraction_engine.py:40`、`:74`、`:92` 与 `clients/agents/runtime_factory.py:119` 的 `except Exception: pass`，改为分类异常 + 日志 + 可追踪错误
- [x] 清理 `constants.py` 14 处实现说明性注释（配置命名已自解释）；TODO 所引 `services/neo4j_repository.py` 为笔误，实际 `repositories/neo4j_repository.py` 的 `_merge_entity_tx` 说明注释已删除
- [x] 将 `services/tts_audio_quality.py`、`services/tts_post_processing.py` 的 `numpy` 依赖下沉到基础设施适配器
- [x] 将 `services/tts_preference_store.py` 的 `pymongo` 与硬编码库名 `"rag_system"`（`:39`）改为注入仓储/配置
- [x] 复核主依赖 `bitsandbytes`（传递依赖 `torch>=2.3`）在全新环境安装时可能拉取默认 PyPI torch 的问题：移入 `[quantization]` extra，新增 `scripts/check_torch_build.py` 与 `scripts/install.sh` 提供统一 constraints 与安装顺序预检
- [x] 为 TTS 服务定义接口，替换 `evaluator: Any`（`ITTSQualityEvaluator`）、`notifier: Any`（`ITTSNotifier`）、`collection: Any`（`ITTSCollection` 协议）
- [x] `controllers/tts_controller.py` 改为依赖注入：`main.py` lifespan 组合 `TTSService`/`StyleTTS2Adapter` 并调用 `set_tts_service`；控制器不再直接实例化适配器，未注入时返回 503
- [x] 拆分 `services/evolution_dispatcher.py`：`_apply_adjustments`(37→`_dispatch_dominant_adjustments`/`_record_adjustment_history`)、`_aggregate_signals`(29→`_count_target_totals`)、`_apply_m3`(26→`_increment_chunk_importance`)、`_adjust_agent_preferences`(24→`_adjust_agent_type_preference`/`_adjust_skill_preferences`)，全部 ≤20 行
- [x] 拆分 `repositories/mongodb_repository.py`：`save_document` 改 `(doc_id, content, title, source, **fields)` + `_build_document_record`；`save_chunk` + `_build_chunk_record(spec, now)`；`__init__` + `_qualified_uri`/`_authenticated_uri`；`repositories/neo4j_repository.py` 3 个事务函数提取 `_RELATION_NODES_QUERY`/`_RELATION_MERGE_QUERY`/`_SUBGRAPH_QUERY`/`_LTM_MERGE_QUERY` 常量与 `_ensure_relation_nodes`/`_ltm_params`/`_subgraph_nodes`/`_subgraph_edges`，全部 ≤20 行
- [x] 拆分 `interfaces/molt_types.py::__post_init__`（25 行）为 `_validate_probabilities` / `_validate_steps` 两个函数
- [x] 统一 TTS 命名：`TTSTAlertService`、`TTSSpeechQualityGateway`、`TTSSRegressionSuite`、`TTSTrendStatistics` 改为一致的 `TTS...` 前缀
- [x] 消除 `utils/mcp_loader.py:58` 的模块级全局 `_connections` Dict，改为 `MCPConnectionRegistry` 实例封装（set/get/list/values/clear）
- [x] 复核 `bootstrap/container.py:230` 全局单例：DI container 生命周期对象（含 override/单例注册表与 shutdown）可接受；真正的泄漏点是 `wire_routes` 直取 `inference_svc._engine`/`_current_model` 私有属性，改为 `domain/inference/services.py` 新增公开 `current_engine` 属性后经公开接口获取
- [x] 将 `services/tts_alerts.py:11-13`、`services/tts_audio_quality.py:12`、`services/tts_regression_suite.py:29`、`services/tts_post_processing.py` 的魔法数字改为配置项
- [x] 为 `/tts`、`/tts/clone` 增加输入校验（空文本、空 alias、base64 格式与大小）
- [x] 修正 `tts_estimate/main.py:74` 循环内 f-string 日志，改为结构化字段并在循环外汇总

### 6.4 质量门禁（防止回归）

- [x] CI 增加单文件 ≤500 行、单函数 ≤20 行检查（`scripts/quality_gate.py`，基线容错：新违规即失败、已记录项不允许增长）
- [x] CI 增加禁止 `print(`、`except ...: pass`、业务层直接导入 `pymongo` / `numpy` / `redis` 的检查（同上 quality gate；`services`/`controllers`/`agents`/`application`/`domain`）
- [x] CI 增加禁止新增解释性注释的检查（`scripts/quality_gate.py` 的 `explanatory_comments` 规则，排除工具指令与契约/类型文档，基线容错）
- [x] CI 增加 TODO 勾选与测试结果一致性检查（`scripts/todo_guard.py` + CI `--junitxml` 报告断言：勾选任务对应测试必须为绿）
- [x] 复核后同步更新第 1–5 节勾选状态，并在本节记录整改结果（证据链与 3 项回退见 §6.5）

### 6.5 整改结果记录（2026-10-30）

- §6.2 失败测试修复：已完成 `tests/test_mcp_env_loading.py`、`tests/test_mcp_protocol.py`、`tests/test_molt_integration.py`、`tests/test_molt_scheduler.py`、`tests/test_skills_env_loading.py`、`tests/test_version_manager.py` 的失败用例全部修复并通过；另修复 MCP 侧失败用例（agent 默认工具注册、`execute` 参数、`dead process` 注册）。
- §6.1 MCP stdout 健壮性：`utils/mcp_protocol.py` 阻塞式 `stdout.read` 改为 `read1()`（支持大 JSON/无换行）；`utils/mcp_loader.py` 日志 `extra={"name": ...}` 与 `LogRecord.name` 冲突改为 `extra={"connection": ...}`；loader 对已退出进程不再注册/降级注册 tools。
- §6.3 依赖下沉：`services/tts_audio_quality.py`、`services/tts_post_processing.py` 移除业务层 `numpy`；`services/tts_preference_store.py` 改为注入 collection/database。
- §4 TTS 真实接入：新增 `model_providers/tts/style_tts2_engine.py`（`StyleTTS2Engine` 懒加载 checkpoint、真实推理、句子级流式 WAV、speaker embedding）、`infrastructure/tts/speaker_repository.py`、`interfaces/tts_types.py`；重写 `model_providers/tts/style_tts2_adapter.py`（真实 streamer/cloner、Redis embedding、Mongo 持久化、偏好注入）；`services/tts_service.py` 编排监控/质量/趋势/报警/反馈/回归；`controllers/tts_controller.py` 改 `StreamingResponse(audio/wav)` + 校验 + `/tts/*` 监控接口。
- §2.1 MOLT 生产接入：`main.py` lifespan 中实例化 `MoltController`（注入 Mongo/Neo4j/Decay/Abstraction/Version/Feedback 依赖）并启动 `MoltScheduler`，任务经 `AsyncTaskScheduler` 提交执行（`LEARNING` 关闭时跳过）。
- §6.3 TTS 命名与配置：重命名 `TTSTAlertService`→`TTSAlertService`、`TTSSpeechQualityGateway`→`TTSQualityGateway`、`TTSSRegressionSuite`→`TTSRegressionSuite`；WER/DNSMOS/similarity/告警数/采样率/最大时长阈值迁移到 `constants`（`TTS_*`）；`tts_estimate/main.py` 循环内 f-string 日志改为结构化字段。
- §6.3 TTS 服务接口：`interfaces/tts_types.py` 新增 `ITTSQualityEvaluator` / `ITTSNotifier` 端口，质量网关与告警服务改用端口类型（替代 `Any`）。
- §6.3 `rag.py`：`command_line_rag` 的 print 改为 logger；ES 存在性检查的裸 `except Exception: pass` 改为分类日志。
- §6.3 依赖注入收尾：`interfaces/tts_types.py` 新增 `ITTSCollection` 协议替换 `collection: Any`；`main.py` lifespan 作为 TTS 组合根注入 `TTSService`（adapter/speaker repo/preference store），控制器未注入时返回 503 而非兜底实例化。
- §6.4 质量门禁：新增 `scripts/quality_gate.py` + `scripts/quality_baseline.json`（基线容错，只拦新增违规与已记录项增长）与 `.github/workflows/ci.yml`（quality-gate + targeted-tests 两个 job）；定向测试 job 固定运行验证通过的 21 个测试文件；CI 安装顺序为先 CPU torch 再 `pip install -e .`，避免拉取默认 PyPI torch 覆盖 ROCm/CUDA 构建。
- §6.2 后续跟进，2026-11-04：全量失败从 19 降至 16——`pyproject.toml` 增加 `[tool.pytest.ini_options] asyncio_mode=auto` 修复 `tests/test_error_handler.py` / `tests/test_private_context.py`；`model_providers/model_provider.py` 构造期 `asyncio.create_task` 无运行事件循环时崩溃改为惰性启动（`_ensure_scheduler_task`），修复 `tests/test_agent_system.py`。剩余 16 个失败均为 Neo4j 未运行 / API 服务器未启动的连接类错误。
- §6.3 后续跟进，2026-11-04：`main.py` lifespan 拆分为 `init_molt_scheduler` / `init_tts_service` < 20 行；MOLT 依赖装配下沉到 `bootstrap/molt_factory.py`（`build_molt_controller` + `_optional_repo`）。质量门禁（`scripts/quality_gate.py` + 基线）与全量测试维持通过（143 passed / 16 failed，失败均为 Neo4j 或 API 服务未启动）。
- §6.3 后续跟进（第 2 批）：`rag.py`→`rag/` 包拆分完成（`splitting`/`document_loader`/`document_ingestion`/`vectorstore_support`/`local_rag`，经 `git mv` 保留历史）；`load_documents` 改为扩展名分发表；超长函数全部拆至 ≤20 行；质量门禁零基线迁移通过；全量测试与拆分前完全一致（143 passed / 16 failed）。
- §6.3 后续跟进（第 3 批）：`evolution_dispatcher`/`mongodb`/`neo4j` 超长函数拆分完成（全部 ≤20 行，质量门禁零基线迁移）；`save_document` 改 kwargs 形态；`wire_routes` 私有属性泄漏经 `InferenceDomainService.current_engine` 公开属性消除；`quality_gate.py` 语法解析失败不再静默跳过（计入 parse_errors 违规）；全量测试维持 143 passed / 16 failed（环境类）。
- §6.3 `clients/agents/runtime_factory.py`：`load_from_env` 失败改为 `logger.exception` 分类处理。

- §6.1 DNSMOS 复核，2026-11-05：按官方 `microsoft/DNS-Challenge` 的 `dnsmos_local.py` 重写 `tts_estimate/infrastructure/dnsmos.py`——16 kHz 波形、9.01 s/144160 采样窗口、`[1, 144160]` float32 输入、1 s hop、短音频重复补齐、首个输出 `[1, 3]` 按 SIG/BAK/OVR 取值并做官方 polyfit 校准，领域映射 `overall=OVR / signal=SIG / background=BAK`；`onnxruntime`/`librosa` 改为按需导入、`session` 可注入，新增 `tests/test_dnsmos_inference.py`（7 项）与 `tts_estimate/README.md` 推理契约表。
- §6.3 bitsandbytes 复核，2026-11-05：`bitsandbytes>=0.43.0` 从主依赖移到 `[quantization]` extra；新增 `scripts/check_torch_build.py`（宿主 torch 预检 + 生成 torch/torchaudio/torchvision 版本 constraints）与 `scripts/install.sh`（预检 → `-c` 安装 → 复核），`scripts/setup_styletts2.sh` 复用同一生成器；CI 安装步骤加 `-c torch-constraints.txt`。
- §6.4 质量门禁，2026-11-05：`scripts/quality_gate.py` 新增 `explanatory_comments` 规则（tokenize 提取注释，排除 shebang/编码/工具指令/契约类型文档/纯分隔线），基线容错只拦新增与增长，`scripts/quality_baseline.json` 重新生成；新增 `scripts/todo_guard.py`（`--list` / `--check` / `--report`）在 CI 校验勾选任务引用的测试文件存在且 JUnit 报告全绿。
- §6.4 第 1–5 节勾选同步，2026-11-05（证据链）：§4 生成/克隆/流式/偏好已落地（`style_tts2_adapter.py:120-167`、`tts_controller.py:106`、`main.py:110-118`）；后处理经 `style_tts2_engine.py:106` 的 `postprocess=process_samples` 落地，`TTSPostProcessor` 外观类仍无调用点（保留勾选，列为清理项）；§2.1 `MoltScheduler` 经 `main.py:170` 接入，`abstraction_engine.py:27` 的 `memory_repo` 别名已去除；§1 MCP `read1()` 见 `utils/mcp_protocol.py:194-197`。回退为 `[ ]` 的 3 项：§4 质量网关（`TTSService.quality_gateway` 默认 `None` 且 `main.py` 未注入，生产链路无 evaluator）、§5 抽样接入评估流水线（`_record_request` 只记录时长/重复）、§5 每日回归/版本对比（`/tts/regression` 为手动接口，`TTSRegressionSuite.compare` 未被调用且无每日调度）。
- 验证，2026-11-05：便携 Python 3.12 运行 `scripts/quality_gate.py`（通过）与 `scripts/todo_guard.py --check`（6 个引用全部解析）；新增测试需在 WSL 环境执行 `python -m pytest -q tests/test_dnsmos_inference.py tests/test_todo_guard.py`。
