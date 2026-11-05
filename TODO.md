# TODO — 未完成任务列表

> 整理于 2026-11-05：依据 `audit-2026-11-05.md` 移除 58 项已核验完成的任务；其余 31 项 PARTIAL + 1 项 NO_EVIDENCE 回退为待办并挂子任务，另新增 3 项 P0 运行期缺陷。
> 本文件只保留未完成的任务；已移除内容与历史整改记录见 git 历史 `44d1d81`。

## 2. MOLT 类脑记忆系统（来源：MOLT_BRAIN_PLAN.md + MOLT_BRAIN_SPEC.md）

### 2.1 MOLT 工程化打磨（补充任务）

- [ ] **打通 MoltScheduler 与现有任务调度器**：查找并接入项目实际的统一调度器（若无则新增最小适配），配置 decay/abstraction/evaluation 周期、失败重试、并发控制，并在 `LEARNING` 开关下安全注册
  - [ ] 子任务（`§2.1-7` / PARTIAL）：`LEARNING` 默认 false，且 `current_memory_repo()` 依赖仅在 `PRE_INIT_RAG=true` 时创建的 `backend_globals.local_rag`，整条链可能为空；**另见 §7 P0-1**（`main.py:93` 的 `NameError` 使 MOLT 生产链路永不启动）。修 P0-1 后补启动验证与开关矩阵测试。
## 3. RAG 架构（来源：RAG_SPEC.md）

- [ ] 补齐 `infrastructure/rag/milvus_store.py` 与 `infrastructure/rag/local_rag_service.py::MilvusVectorStore` 的占位实现（`add_documents` 仅返回 id、`similarity_search` 返回空）
  - [ ] 子任务（`§3-3` / PARTIAL）：`MilvusVectorStore` / `LocalRAGService` 只注册在 `bootstrap/container.py`，`wire_routes()` 不注入 `rag_service`，唯一外部调用点是 `tests/test_container_lazy.py`；主链路仍用 `rag.LocalRAG` + `langchain_milvus`。需决策保留哪条链路：接入生产或清理死代码。
- [ ] 增加 checksum 变更触发重建与失败状态重试
  - [ ] 子任务（`§3-5` / PARTIAL）：`rag/document_ingestion.py:39-45` 的 checksum 比对、重建与 `status` 断点续跑已落地（`services/ingestion_checkpoint.py`），但「失败状态重试」无独立状态机/重试入口，仅 `_execute_step` 记日志。需补失败状态持久化与可重跑/重试入口。

## 4. TTS 情感语音系统（来源：TTS_SPEC.md）

- [ ] 将质量网关（WER/DNSMOS/Similarity 自动拦截）接入 TTS 生成链路
  - 现状证据：`TTSService.quality_gateway` 默认 `None` 且 `main.py::init_tts_service` 未注入（`services/tts_service.py:30,42`、`main.py:111-117`）；`interfaces/tts_types.py:17` 的 `ITTSQualityEvaluator` 无生产实现，`tts_estimate/main.py:25` 的组合根只服务独立评估项目；未注入 evaluator 时 `TTSQualityGateway.evaluate` 返回 `rejected` 会把全部请求拒掉；`/tts` 未传 `reference_audio`（`controllers/tts_controller.py:103-105`）；`_record_request` 在流全部 yield 之后才执行（`services/tts_service.py:78`），抛 `TTSQualityRejected` 已无法阻断本次响应。
  - 待人为抉择 A（拦截时机/强度）：生成前预检 / 逐句生成时拦截并流式回退 / 仅事后评估+告警不阻断。流式首包 <500ms 约束下是否允许同步评估，需明确。
  - 待人为抉择 B（参考音频来源）：克隆会话的 speaker embedding 缓存 / 请求携带参考音频 / 固定基准音频库；并决定无参考音频时是跳过网关还是降级为仅 DNSMOS。
  - 待人为抉择 C（评估器与成本）：是否复用 `tts_estimate` 的 Whisper + DNSMOS ONNX + ECAPA（每次抽样推理成本较高），或先用轻量代理指标；是否允许按需懒加载模型。
  - 待人为抉择 D（阈值与降级）：WER/DNSMOS/Similarity 的拦截阈值取值、`should_block` 的判定逻辑，以及 evaluator 不可用/超时时是放行、告警还是拒绝（当前为拒绝）。
  - 待人为抉择 E（失败语义）：拦截后返回的 HTTP 状态码/错误体、是否触发异步重生成，以及流式已发出音频时如何向客户端表达质量不合格。
- [ ] 实现流式输出（首包延迟 <500ms）
  - [ ] 子任务（`§4-7` / PARTIAL）：`StreamingResponse` 与 `elapsed_ms` 计时存在，但首包是整句 10 步 diffusion 合成后的完整 WAV，无首包延迟测量或断言（测试只断言 `>=0`）。需定义首包口径并加 SLA 断言，或把描述修正为实际能力。
- [ ] 实现后处理呼吸感与随机 jitter
  - [ ] 子任务（`§4-9` / PARTIAL）：`style_tts2_engine.py:113` 只传 `(wave, emotion_params)` 两参，`process_samples` 的 `seed` 恒为 `None`，`audio_processing.py:28-29` 直接 return → 只有呼吸感生效，jitter 空转。需传入 seed 并打通 jitter 分支。

## 5. TTS 质量评估与线上监控（来源：TTS_QUALITY_ESTIMATE.md）

- [ ] 生产监控抽样（5%~10% 请求）接入评估流水线
  - 现状证据：抽样钩子已存在且默认 5%（`services/tts_production_monitor.py:9`、`services/tts_service.py:123-124`），但记录的 `result` 只有 status/时长/时长异常/重复片段/文本（`services/tts_service.py:115-121`），不含 WER/DNSMOS/Similarity；`tts_estimate` 的评估器从未在 `TTSService` 中实例化，故 `metrics()` 里的三项质量统计（`services/tts_service.py:86-88`）在生产中恒为空；`TTSSamplingMonitor._results` 仅为进程内 list，无持久化。
  - 待人为抉择 A（抽样率与策略）：固定 5% / 10% / 可配置区间，是否按用户、speaker、时段分层抽样，是否对失败请求 100% 采样。
  - 待人为抉择 B（评估深度）：全量三项指标 / 仅 DNSMOS / 按请求类型分级；以及 WER 所需参考文本、Similarity 所需参考音频从哪里取。
  - 待人为抉择 C（同步 vs 异步）：当前在请求路径内同步执行，是否改为后台任务队列 + 结果落库，避免拖慢音频响应；队列满载时的丢弃策略。
  - 待人为抉择 D（结果落点与保留期）：写回 `TTSTrendStatistics` / `TTSAlertService` / 采样记录 / MongoDB 时序集合，保留窗口与脱敏要求（当前文本原文会进入内存记录）。
  - 待人为抉择 E（成本与配额）：每日评估次数上限、模型推理并发上限、超预算时的降级策略。
- [ ] 每日统计、滚动均值、P90/P95、异常比例
  - [ ] 子任务（`§5-1` / PARTIAL）：统计方法真实且已接入 `metrics()`，但 `_record_request` 的 `result` 不含 wer/dnsmos/similarity → 三项质量统计生产恒为 0；仅进程内 `deque(maxlen=1000)`，无「每日」聚合与持久化。需补指标来源、每日聚合与持久化。
- [ ] 报警阈值与通知（WER / DNSMOS / Speaker Similarity / 失败率）
  - [ ] 子任务（`§5-2` / PARTIAL）：`should_alert` 四阈值真实，但 `notifier` 未注入、生产仅在时长/重复异常时 `check`，三个质量阈值分支恒不触发。需注入生产 notifier 并覆盖质量分支。
- [ ] 固定测试句集 + 每日回归测试 + 版本对比曲线
  - 现状证据：固定句集已有 3 句英文（`services/tts_regression_suite.py:11-15`），但只有手动接口 `POST /tts/regression`（`controllers/tts_controller.py:138-141`，限流 2/min），无任何每日调度（`main.py:170` 只注册 MOLT 任务）；`TTSService.run_regression` 只返回 `duration_seconds`/`bytes`（`services/tts_service.py:94-107`），未调用 `TTSRegressionSuite.run()`，`_runs` 历史不写入；`compare()`（`services/tts_regression_suite.py:40`）唯一调用点是 `tests/test_tts_regression_suite.py:18`；`version` 为随机 `uuid4()[:8]`，无真实版本与持久化，无法形成曲线。
  - 待人为抉择 A（句集内容）：是否保留现有 3 句英文，改为中文/中英混合、覆盖多情绪与长难句、数字与专有名词；句集数量与存放位置（代码常量 / 配置文件 / 数据库）。
  - 待人为抉择 B（调度方式）：复用 `AsyncTaskScheduler` / `TaskScheduler` / 独立 cron / CI 定时任务；执行时间、时区、并发与失败重试策略。
  - 待人为抉择 C（回归指标）：在时长/字节之外是否加入 WER/DNSMOS/Similarity、首包延迟、失败率；阈值与判定为回归的标准。
  - 待人为抉择 D（版本标识）：用 checkpoint 名 / git sha / 模型+配置快照 替换随机 uuid；版本元数据从哪读取。
  - 待人为抉择 E（持久化与曲线）：结果存 MongoDB / 时序库 / 静态报表，对比基线（上一版本 vs 固定基线）、保留窗口、曲线展示方式与回归告警阈值。
  - 待人为抉择 F（手动接口）：是否保留 `POST /tts/regression` 及其 2/min 限流，还是改为只由调度触发。
- [ ] 为 `tts_estimate` 增加批量评估与 CLI 参数入口（当前 `main.py` 使用 mock 路径）
  - [ ] 子任务（`§5-6` / PARTIAL）：**另见 §7 P0-2**——`run_evaluation` 返回 JSON 字符串而 `main()` 调用 `report.get(...)`，CLI 批量评估必崩 `AttributeError`。需修 P0-2 并补 CLI 冒烟测试。

## 6. 代码原则审计整改（2026-10-30）

> 说明：本节按「勾选真实性复核 / 失败测试 / 代码原则 / 质量门禁」分类。2026-11-05 全量核查后，无缺陷的勾选项已从本文件移除，仍有缺陷的项已回退为 `- [ ]` 并挂子任务。

### 6.1 TODO 勾选真实性复核

- [ ] 复核 §4 质量网关接入：`TTSSpeechQualityGateway` 除测试外无引用，未进入 TTS 生成链路
  - [ ] 子任务（`§6.1-5` / PARTIAL）：`TTSQualityGateway` 默认 `None` 且 `main.py::init_tts_service` 不注入，`ITTSQualityEvaluator` / `ITTSNotifier` 无生产实现。需与 §4「质量网关接入」的待抉择任务合并推进。
- [ ] 复核 §4 后处理接入：`TTSPostProcessor` 除测试外无引用，未进入 TTS 生成链路
  - [ ] 子任务（`§6.1-6` / NO_EVIDENCE）：全库找不到 `TTSPostProcessor` 的生产调用点；实际后处理走 `style_tts2_adapter.py:72` 的 `postprocess=process_samples`。需接入该外观类或删除死代码。
- [ ] 复核 §5 生产监控/统计/报警/反馈/回归：`TTSSamplingMonitor`、`TTSTrendStatistics`、`TTSTAlertService`、`TTSFeedbackMonitor`、`TTSAudioQualityService`、`TTSSRegressionSuite` 均只被测试引用，未接入 `/tts` 或定时任务
  - [ ] 子任务（`§6.1-7` / PARTIAL）：部分组件现已在 `TTSService` 构造，但评估流水线、通知实现与每日回归调度仍缺生产接入点。需逐项核实并补可复现证据链。
- [ ] 复核 §2.1 MoltScheduler 接入：`services/molt_scheduler.py` 未被任何生产代码实例化，`MoltController` / `AbstractionEngine` / `VersionManager` / `DecayScheduler` 也仅出现在测试中
  - [ ] 子任务（`§6.1-9` / PARTIAL）：**另见 §7 P0-1**。修 P0-1 后重新核对实例化与启动证据，再决定是否重新勾选。

### 6.2 失败测试修复

- [ ] CI 增加定向测试步骤（`.github/workflows/ci.yml` 运行 21 个已验证通过的定向测试文件），禁止以「环境问题」跳过上述失败用例
  - [ ] 子任务（`§6.2-8` / PARTIAL）：`ci.yml` 注释明写 "green set; env-dependent tests excluded"，只跑 23 个定向文件，无全量 pytest；「禁止跳过」无制度保障。需增加全量 pytest job，或改为可复现的跳过白名单 + 到期复核。

### 6.3 代码原则整改

- [ ] 删除 `services/abstraction_engine.py:40`、`:74`、`:92` 与 `clients/agents/runtime_factory.py:119` 的 `except Exception: pass`，改为分类异常 + 日志 + 可追踪错误
  - [ ] 子任务（`§6.3-6` / PARTIAL）：抽查未发现字面 `except Exception: pass` 残留，但「分类异常 + 日志 + 可追踪错误」未逐点复核（尤其 `clients/agents/runtime_factory.py:119`）。需逐点复核并补回归测试。
- [ ] 清理 `constants.py` 14 处实现说明性注释（配置命名已自解释）；TODO 所引 `services/neo4j_repository.py` 为笔误，实际 `repositories/neo4j_repository.py` 的 `_merge_entity_tx` 说明注释已删除
  - [ ] 子任务（`§6.3-7` / PARTIAL）：`constants.py` 仍留 2 处说明性注释。需清理剩余 2 处。
- [ ] 将 `services/tts_audio_quality.py`、`services/tts_post_processing.py` 的 `numpy` 依赖下沉到基础设施适配器
  - [ ] 子任务（`§6.3-8` / PARTIAL）：`services/tts_post_processing.py:4` 仍 `import numpy as np`（已登记进 `quality_baseline.json`，故门禁不拦）。需下沉到基础设施适配器。
- [ ] 将 `services/tts_preference_store.py` 的 `pymongo` 与硬编码库名 `"rag_system"`（`:39`）改为注入仓储/配置
  - [ ] 子任务（`§6.3-9` / PARTIAL）：`services/tts_preference_store.py:18` 仍 lazy `from pymongo import MongoClient`（同样已进基线）。需改为注入 collection/database。
- [ ] 拆分 `repositories/mongodb_repository.py`：`save_document` 改 `(doc_id, content, title, source, **fields)` + `_build_document_record`；`save_chunk` + `_build_chunk_record(spec, now)`；`__init__` + `_qualified_uri`/`_authenticated_uri`；`repositories/neo4j_repository.py` 3 个事务函数提取 `_RELATION_NODES_QUERY`/`_RELATION_MERGE_QUERY`/`_SUBGRAPH_QUERY`/`_LTM_MERGE_QUERY` 常量与 `_ensure_relation_nodes`/`_ltm_params`/`_subgraph_nodes`/`_subgraph_edges`，全部 ≤20 行
  - [ ] 子任务（`§6.3-14` / PARTIAL）：**另见 §7 P0-3**——`save_document` 未把 `title` / `content` / `source` 传入 `_build_document_record`，正常调用必抛 `KeyError`；另 `repositories/neo4j_repository.py:143-144` 有重复 `@staticmethod`。需修 P0-3、清理重复装饰器、补 `save_document` 回归测试。
- [ ] 统一 TTS 命名：`TTSTAlertService`、`TTSSpeechQualityGateway`、`TTSSRegressionSuite`、`TTSTrendStatistics` 改为一致的 `TTS...` 前缀
  - [ ] 子任务（`§6.3-16` / PARTIAL）：4 项中只完成 3 项；`TTSTrendStatistics`（`services/tts_trend_statistics.py:9`）与 `TTSSamplingMonitor`（`services/tts_production_monitor.py:8`）命名仍未统一。需统一剩余命名并同步所有引用。
- [ ] 消除 `utils/mcp_loader.py:58` 的模块级全局 `_connections` Dict，改为 `MCPConnectionRegistry` 实例封装（set/get/list/values/clear）
  - [ ] 子任务（`§6.3-17` / PARTIAL）：已封装为 `MCPConnectionRegistry`，但 `utils/mcp_loader.py:85` 仍保留模块级 `_connections = MCPConnectionRegistry()` 单例。需改为显式注册与生命周期管理。
- [ ] 修正 `tts_estimate/main.py:74` 循环内 f-string 日志，改为结构化字段并在循环外汇总
  - [ ] 子任务（`§6.3-21` / PARTIAL）：**另见 §7 P0-2**——日志汇总实际仍在循环内，未改为循环外汇总。需修 P0-2 时一并处理。

### 6.4 质量门禁（防止回归）

- [ ] CI 增加禁止 `print(`、`except ...: pass`、业务层直接导入 `pymongo` / `numpy` / `redis` 的检查（同上 quality gate；`services`/`controllers`/`agents`/`application`/`domain`）
  - [ ] 子任务（`§6.4-2` / PARTIAL）：`scripts/quality_gate.py` 的基线容错把既有违规合法化，`--update-baseline` 可一键绕过。需加基线变更审查（保护分支/PR 审批）或收紧绕过条件。
- [ ] CI 增加 TODO 勾选与测试结果一致性检查（`scripts/todo_guard.py` + CI `--junitxml` 报告断言：勾选任务对应测试必须为绿）
  - [ ] 子任务（`§6.4-4` / PARTIAL）：`todo_guard.py` 只抓勾选行里显式反引号引用的 `tests/*.py`，全库仅 6 个文件被关联，§6.1/6.3/6.4 共 38 个勾选项无任何测试约束。需扩展引用解析或引入显式映射表；另 `scripts/todo_guard.py:88` 为 `return 0 if paths else 1`，本文件现无 `- [x]` 行使 `--check` 因 0 引用退出 1，需改为覆盖待处理项关联测试或增加 `--allow-empty` 并同步 CI。
- [ ] 复核后同步更新第 1–5 节勾选状态
  - [ ] 子任务（`§6.4-5` / PARTIAL）：核查发现多处原记录与代码不符，历史整改记录已移除；各回退项需补可复现证据后再重新勾选。

## 7. P0 运行期缺陷（2026-11-05 核查确认，优先修复）

- [ ] P0-1 MOLT 生产链路永不启动：`main.py:93` 调用 `current_memory_repo()`，但 `main.py:41` 只导入了 `build_molt_controller`；该函数仅定义于 `bootstrap/molt_factory.py:21`，全项目无其它导入 → 每次 lifespan 的 `init_molt_scheduler()` 必抛 `NameError`，且被 `main.py:97-98` 的 `except Exception` 静默吞掉。
  - [ ] 子任务：补 `current_memory_repo` 导入；把该 `except` 改为 `logger.exception` 保留堆栈；补启动成功的集成断言，再复核 `§2.1-7` / `§6.1-9`。
- [ ] P0-2 `tts_estimate` CLI 批量评估必崩：`run_evaluation()`（`tts_estimate/main.py:47`）返回 `json.dumps(...)` 字符串，`main()`（`:95-96`）却调用 `report.get("status")` / `report.get("wer")` → `AttributeError`；且 §6.3-21 声称的「日志循环外汇总」实际仍在循环内。
  - [ ] 子任务：让 `run_evaluation` 返回 dict（或在 `main` 中 `json.loads`），日志汇总移到循环外；补 CLI 冒烟测试，再复核 `§5-6` / `§6.3-21`。
- [ ] P0-3 `save_document` 必抛 `KeyError`：`repositories/mongodb_repository.py:51-65` 的 `save_document(doc_id, content, title, source, **fields)` 只把 `**fields` 传给 `_build_document_record`，后者 `:64-65` 直接索引 `fields["title"]` / `fields["content"]`；而二者是显式形参，不会进入 `**fields`。
  - [ ] 子任务：把 `title` / `content` / `source` 一并传入 `_build_document_record`；补 `save_document` 回归测试，再复核 `§6.3-14`。
