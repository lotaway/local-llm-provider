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
- [x] 将质量网关（WER/DNSMOS/Similarity 自动拦截）接入 TTS 生成链路
- [x] 实现后处理呼吸感与随机 jitter

## 5. TTS 质量评估与线上监控（来源：TTS_QUALITY_ESTIMATE.md）

- [x] 生产监控抽样（5%~10% 请求）接入评估流水线
- [x] 每日统计、滚动均值、P90/P95、异常比例
- [x] 报警阈值与通知（WER / DNSMOS / Speaker Similarity / 失败率）
- [x] 生成失败率、音频时长异常、重复片段检测
- [x] 用户反馈分布（差评 / 投诉 / 重新生成次数）
- [x] 固定测试句集 + 每日回归测试 + 版本对比曲线
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
- [ ] 复核 §5 DNSMOS 真实推理：`tts_estimate/infrastructure/dnsmos.py::_infer_frame` 的输入形状 `[frame]` 与输出索引需按官方模型确认，当前实现可能无法得到有效分数
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
- [ ] CI 增加定向测试步骤，禁止以“环境问题”跳过上述失败用例

### 6.3 代码原则整改

- [ ] 拆分 `rag.py`（1293 行）：按文档加载、分块、向量库、图抽取、ES、查询编排拆分为独立模块，目标单文件 ≤500 行、单函数 ≤20 行
- [ ] 清理 `rag.py` 的 43 处注释、3 处 `print`（`:1275`、`:1283`、`:1287`）及 `except: pass`（`:858`）
- [ ] 降低 `rag.py::load_documents`（`:558`，154 行）的嵌套深度（当前 14 层），并消除 `:578` 起的 9 级 `if/elif` 链
- [ ] 拆分 `rag.py` 超长函数：`build_vectorstore`、`_sync_get_or_create_vectorstore`、`_async_extract_graph`、`init_rag_chain`、`_select_text_splitter` 等
- [ ] 拆分 `services/abstraction_engine.py::abstract_to_ltm`（56 行），将 Neo4j/Milvus/ES 写入移至独立仓储方法，消除“返回值 + 副作用”
- [x] 删除 `services/abstraction_engine.py:40`、`:74`、`:92` 与 `clients/agents/runtime_factory.py:119` 的 `except Exception: pass`，改为分类异常 + 日志 + 可追踪错误
- [ ] 清理 `constants.py` 14 处注释及 `services/neo4j_repository.py:45-79` 等实现说明性注释，用类型、命名、拆函数替代
- [x] 将 `services/tts_audio_quality.py`、`services/tts_post_processing.py` 的 `numpy` 依赖下沉到基础设施适配器
- [x] 将 `services/tts_preference_store.py` 的 `pymongo` 与硬编码库名 `"rag_system"`（`:39`）改为注入仓储/配置
- [ ] 复核主依赖 `bitsandbytes`（传递依赖 `torch>=2.3`）在全新环境安装时可能拉取默认 PyPI torch 的问题；首次安装前应先安装宿主 ROCm/CUDA 构建，或提供统一 constraints
- [ ] 为 TTS 服务定义接口，替换 `evaluator: Any`、`notifier: Any`、`collection: Any`
- [ ] `controllers/tts_controller.py` 改为依赖注入 `StyleTTS2Adapter` 抽象，不在路由内直接实例化
- [ ] 拆分 `services/evolution_dispatcher.py::_apply_adjustments`（37 行）、`_aggregate_signals`（29 行）、`_apply_m3`（26 行）
- [ ] 拆分 `repositories/mongodb_repository.py::save_document`（43 行）、`save_chunk`（30 行）及 `repositories/neo4j_repository.py` 的 3 个 26–29 行事务函数
- [ ] 拆分 `interfaces/molt_types.py::__post_init__`（25 行）为范围校验与步长校验两个函数
- [x] 统一 TTS 命名：`TTSTAlertService`、`TTSSpeechQualityGateway`、`TTSSRegressionSuite`、`TTSTrendStatistics` 改为一致的 `TTS...` 前缀
- [x] 消除 `utils/mcp_loader.py:58` 的模块级全局 `_connections` Dict，改为 `MCPConnectionRegistry` 实例封装（set/get/list/values/clear）
- [ ] 复核 `bootstrap/container.py:230` 全局单例是否持有业务状态，必要时拆分生命周期对象
- [x] 将 `services/tts_alerts.py:11-13`、`services/tts_audio_quality.py:12`、`services/tts_regression_suite.py:29`、`services/tts_post_processing.py` 的魔法数字改为配置项
- [x] 为 `/tts`、`/tts/clone` 增加输入校验（空文本、空 alias、base64 格式与大小）
- [x] 修正 `tts_estimate/main.py:74` 循环内 f-string 日志，改为结构化字段并在循环外汇总

### 6.4 质量门禁（防止回归）

- [ ] CI 增加单文件 ≤500 行、单函数 ≤20 行检查
- [ ] CI 增加禁止 `print(`、`except ...: pass`、业务层直接导入 `pymongo` / `numpy` / `redis` 的检查
- [ ] CI 增加禁止新增解释性注释的检查（契约/类型文档除外）
- [ ] CI 增加 TODO 勾选与测试结果一致性检查：勾选任务对应测试必须为绿
- [ ] 复核后同步更新第 1–5 节勾选状态，并在本节记录整改结果

### 6.5 整改结果记录（2026-10-30）

- §6.2 失败测试修复：已完成 `tests/test_mcp_env_loading.py`、`tests/test_mcp_protocol.py`、`tests/test_molt_integration.py`、`tests/test_molt_scheduler.py`、`tests/test_skills_env_loading.py`、`tests/test_version_manager.py` 的失败用例全部修复并通过；另修复 MCP 侧失败用例（agent 默认工具注册、`execute` 参数、`dead process` 注册）。
- §6.1 MCP stdout 健壮性：`utils/mcp_protocol.py` 阻塞式 `stdout.read` 改为 `read1()`（支持大 JSON/无换行）；`utils/mcp_loader.py` 日志 `extra={"name": ...}` 与 `LogRecord.name` 冲突改为 `extra={"connection": ...}`；loader 对已退出进程不再注册/降级注册 tools。
- §6.3 依赖下沉：`services/tts_audio_quality.py`、`services/tts_post_processing.py` 移除业务层 `numpy`；`services/tts_preference_store.py` 改为注入 collection/database。
- §4 TTS 真实接入：新增 `model_providers/tts/style_tts2_engine.py`（`StyleTTS2Engine` 懒加载 checkpoint、真实推理、句子级流式 WAV、speaker embedding）、`infrastructure/tts/speaker_repository.py`、`interfaces/tts_types.py`；重写 `model_providers/tts/style_tts2_adapter.py`（真实 streamer/cloner、Redis embedding、Mongo 持久化、偏好注入）；`services/tts_service.py` 编排监控/质量/趋势/报警/反馈/回归；`controllers/tts_controller.py` 改 `StreamingResponse(audio/wav)` + 校验 + `/tts/*` 监控接口。
- §2.1 MOLT 生产接入：`main.py` lifespan 中实例化 `MoltController`（注入 Mongo/Neo4j/Decay/Abstraction/Version/Feedback 依赖）并启动 `MoltScheduler`，任务经 `AsyncTaskScheduler` 提交执行（`LEARNING` 关闭时跳过）。
- §6.3 TTS 命名与配置：重命名 `TTSTAlertService`→`TTSAlertService`、`TTSSpeechQualityGateway`→`TTSQualityGateway`、`TTSSRegressionSuite`→`TTSRegressionSuite`；WER/DNSMOS/similarity/告警数/采样率/最大时长阈值迁移到 `constants`（`TTS_*`）；`tts_estimate/main.py` 循环内 f-string 日志改为结构化字段。
- §6.3 TTS 服务接口：`interfaces/tts_types.py` 新增 `ITTSQualityEvaluator` / `ITTSNotifier` 端口，质量网关与告警服务改用端口类型（替代 `Any`）。
- §6.3 `rag.py`：`command_line_rag` 的 print 改为 logger；ES 存在性检查的裸 `except Exception: pass` 改为分类日志。
- §6.3 `clients/agents/runtime_factory.py`：`load_from_env` 失败改为 `logger.exception` 分类处理。
