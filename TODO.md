# TODO — 未完成任务列表

> 整理于 2026-11-05：依据 `audit-2026-11-05.md` 移除 58 项已核验完成的任务；其余 31 项 PARTIAL + 1 项 NO_EVIDENCE 回退为待办并挂子任务，另新增 3 项 P0 运行期缺陷。
> 本文件只保留未完成的任务；已移除内容与历史整改记录见 git 历史 `44d1d81`。

## 3. RAG 架构（来源：RAG_SPEC.md）

- [ ] 补齐 `infrastructure/rag/milvus_store.py` 与 `infrastructure/rag/local_rag_service.py::MilvusVectorStore` 的占位实现（`add_documents` 仅返回 id、`similarity_search` 返回空）
  - [ ] 子任务（`§3-3` / PARTIAL）：`MilvusVectorStore` / `LocalRAGService` 只注册在 `bootstrap/container.py`，`wire_routes()` 不注入 `rag_service`，唯一外部调用点是 `tests/test_container_lazy.py`；主链路仍用 `rag.LocalRAG` + `langchain_milvus`。需决策保留哪条链路：接入生产或清理死代码。
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
## 5. TTS 质量评估与线上监控（来源：TTS_QUALITY_ESTIMATE.md）

- [ ] 生产监控抽样（5%~10% 请求）接入评估流水线
  - 现状证据：抽样钩子已存在且默认 5%（`services/tts_request_sampler.py:9`、`services/tts_service.py:123-124`），但记录的 `result` 只有 status/时长/时长异常/重复片段/文本（`services/tts_service.py:115-121`），不含 WER/DNSMOS/Similarity；`tts_estimate` 的评估器从未在 `TTSService` 中实例化，故 `metrics()` 里的三项质量统计（`services/tts_service.py:86-88`）在生产中恒为空；`TTSRequestSampler._results` 仅为进程内 list，无持久化。
  - 待人为抉择 A（抽样率与策略）：固定 5% / 10% / 可配置区间，是否按用户、speaker、时段分层抽样，是否对失败请求 100% 采样。
  - 待人为抉择 B（评估深度）：全量三项指标 / 仅 DNSMOS / 按请求类型分级；以及 WER 所需参考文本、Similarity 所需参考音频从哪里取。
  - 待人为抉择 C（同步 vs 异步）：当前在请求路径内同步执行，是否改为后台任务队列 + 结果落库，避免拖慢音频响应；队列满载时的丢弃策略。
  - 待人为抉择 D（结果落点与保留期）：写回 `TTSQualityStatistics` / `TTSAlertService` / 采样记录 / MongoDB 时序集合，保留窗口与脱敏要求（当前文本原文会进入内存记录）。
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
## 6. 代码原则审计整改（2026-10-30）

> 说明：本节按「勾选真实性复核 / 失败测试 / 代码原则 / 质量门禁」分类。2026-11-05 全量核查后，无缺陷的勾选项已从本文件移除，仍有缺陷的项已回退为 `- [ ]` 并挂子任务。

### 6.1 TODO 勾选真实性复核

- [ ] 复核 §4 质量网关接入：`TTSSpeechQualityGateway` 除测试外无引用，未进入 TTS 生成链路
  - [ ] 子任务（`§6.1-5` / PARTIAL）：`TTSQualityGateway` 默认 `None` 且 `main.py::init_tts_service` 不注入，`ITTSQualityEvaluator` / `ITTSNotifier` 无生产实现。需与 §4「质量网关接入」的待抉择任务合并推进。
- [ ] 复核 §5 生产监控/统计/报警/反馈/回归：`TTSRequestSampler`、`TTSQualityStatistics`、`TTSTAlertService`、`TTSFeedbackMonitor`、`TTSAudioQualityService`、`TTSSRegressionSuite` 均只被测试引用，未接入 `/tts` 或定时任务
  - [ ] 子任务（`§6.1-7` / PARTIAL）：部分组件现已在 `TTSService` 构造，但评估流水线、通知实现与每日回归调度仍缺生产接入点。需逐项核实并补可复现证据链。
### 6.4 质量门禁（防止回归）

- [ ] 复核后同步更新第 1–5 节勾选状态
  - [ ] 子任务（`§6.4-5` / PARTIAL）：核查发现多处原记录与代码不符，历史整改记录已移除；各回退项需补可复现证据后再重新勾选。
