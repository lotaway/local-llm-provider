# TODO — 未完成任务列表

> 整理于 2026-11-05：依据 `audit-2026-11-05.md` 移除 58 项已核验完成的任务；其余 31 项 PARTIAL + 1 项 NO_EVIDENCE 回退为待办并挂子任务，另新增 3 项 P0 运行期缺陷。
> 约定（2026-11-10 起）：本文件只保留未完成任务；逐项核验通过的任务直接从本文件移除，证据留在 git 历史与对应测试中。若再出现勾选项，其定向测试须在 `scripts/todo_test_map.json` 与 `.github/workflows/ci.yml` 中登记并保持绿灯。
> 2026-11-07：按「离线筛选 / 抽样告警、异步评估、不阻断响应」决议落地后，§4 首条、§5 首条与 §6.1-5 曾改为 `[x]`。
> 2026-11-09：按「TTS 质量优先、不设 500ms 绝对阈值」决议改写 §4-7；§5-1 每日聚合与 §6.4 门禁红项曾勾选。同批补齐首包耗时观测、分层/失败抽样、每日配额与评估记录持久化。
> 2026-11-10（人工确认）：未完成项均需人工确认后才能实施，人工确认点逐项写在对应任务下：§5 模型部署 A–C、§5 每日回归 A–F、§6.1-7 A–B、§6.4-5 A–B。
> 2026-11-10（完成项复核）：逐条核对 10 个 `[x]`（6 项主任务 + 4 项子任务）。核验通过并移除 5 个：§4-7 流式首包观测（主+子）、§6.1 §4 网关接入复核（主+子）、§6.4 AST 门禁红项。回退为 `[ ]` 并挂子问题 5 个：§4 质量链路接入、§5 抽样监控（主+两条遗留）、§5-1 每日统计，缺陷与复现方式见各条。
> 移除依据：§4-7 由 `services/tts_service.py:62-64` 的 callback → `first_chunk_ms` → `TTSQualityStatistics` / `metrics()` 链路与 `tests/test_tts_quality_integration.py::test_first_chunk_latency_is_recorded_as_an_observation_only` 覆盖；§6.1 的装配结论由组合根与 `test_composition_root_builds_a_service_with_the_observer` 覆盖（仅装配语义，不含质量判定）；§6.4 由 `tests/test_molt_startup.py` 2 passed 覆盖。

## 4. TTS 情感语音系统（来源：TTS_SPEC.md）

- [ ] 将质量评估流水线接入 TTS 生成链路（2026-11-07 决议：离线筛选 / 抽样告警，不在线阻断响应）
  - 已具备（2026-11-10 复核）：`bootstrap/tts_composition.py` 装配 `ProductionTTSQualityEvaluator` → `TTSQualityGateway` → `TTSQualityObserver` → `TTSService.quality_observer`；`/tts` 支持可选 `reference_audio`；`_to_report` 从 `metrics.speaker_similarity` 取数；evaluator 不可用返回 `skipped` 并放行；流式多段 WAV 由 `merge_wav_segments` 合并后再送模型；`tests/test_tts_quality_integration.py` 25 项契约/装配测试绿灯。
  - 子问题 1（时长与异常判定错误）：`model_providers/tts/audio_quality.py:9-10` 按 float32（4 字节/样本）估算时长，但 `StyleTTS2Engine._to_wav` 输出 PCM_16 WAV（`model_providers/tts/style_tts2_engine.py:320-324`）。实测 3.0s 音频被算成 1.5s；`is_duration_anomaly`（`audio_quality.py:13-15`）因此会漏掉最长约 2 倍超限的音频，连带影响 `status`、失败抽样与告警。需按 WAV 头解析采样率/位深/声道（或先 `pcm_from_wav` 再计时），并补真实 WAV 用例——现有 `tests/test_tts_audio_quality.py:23-26` 只喂裸字节，覆盖不到该路径。
  - 子问题 2（请求路径随历史线性退化）：`TTSAudioQualityService._history`（`services/tts_audio_quality.py:14,25`）无上限保存整段音频；`is_duplicate`（`audio_quality.py:22-24`）每次对全部历史逐个 sha256。`_record_request` 在请求协程内同步调用（`services/tts_service.py:121`），历史越长越慢，与「请求路径零阻塞」不符。需改为有界摘要集合（定长 deque + 集合），并把重复比对移出请求路径。
  - 子问题 3（同一请求重复告警且告警无界）：`_record_request` 已执行 `alerts.check([result])`（`services/tts_service.py:131`），观察器评估后又执行 `alerts.check([evaluated])`（`services/tts_quality_observer.py:82`），同一请求可能落两条告警；`TTSAlertService._alerts`（`services/tts_alerts.py:28,38`）只 append 无上限，且保存含 `text` 的整条记录。需明确「一次请求只告警一次」的归属，并给告警存储设上限与脱敏。
  - 原条目「自动拦截」语义经决议不实施：在线阻断、HTTP 错误体、异步重生成不再作为本项验收条件。
  - 已决议 A：仅事后评估 + 告警，不阻断；「首包 <500ms」是 TTS 的 TTFA 体验愿望而非 STT 式硬实时约束。
  - 已决议 B：请求可选携带 `reference_audio`（base64），无参考时跳过 ECAPA 与该阈值，降级为 WER + DNSMOS。
  - 已决议 C：复用 `tts_estimate` 三项模型并按需懒加载，仅对抽样请求推理。
  - 已决议 D：阈值沿用 `constants.py` 的 0.08 / 3.8 / 0.78；evaluator 不可用、推理失败、队列满载均放行并告警。
  - 已决议 E：不阻断响应，无 HTTP 错误语义；失败只写一条 `evaluation_failed` 告警。

## 5. TTS 质量评估与线上监控（来源：TTS_QUALITY_ESTIMATE.md）

- [ ] 生产监控抽样（默认 5%）接入评估流水线
  - 已具备（2026-11-10 复核）：`TTSQualityObserver` 由 `TTSService._record_request` 递交，抽样命中后经有界线程池（`TTS_QUALITY_MAX_WORKERS` / `TTS_QUALITY_MAX_PENDING`，默认 2 / 8）异步跑真实 evaluator，完成后写回 `TTSQualityStatistics` / `TTSRequestSampler` / `TTSAlertService`；队列满载丢弃并记 `tts_quality_evaluation_dropped`；evaluator 缺失或推理失败只告警、不影响响应；25 项契约/装配测试绿灯。
  - 子问题 1（进程内记录无界）：`TTSAlertService._alerts`（`services/tts_alerts.py:28`）与 `TTSRequestSampler._results`（`services/tts_request_sampler.py:29,59`）只 append 不设上限，记录中带 `text`；长跑会持续占用内存并无限期保留原始文本，与「文本脱敏」的意图不一致。需设窗口/上限，并在入内存前做脱敏。
  - 子问题 2（请求路径仍做重活）：同 §4 子问题 2，`has_duplicate` 在请求协程内对全量历史做摘要比对（`services/tts_service.py:121`），抽样监控无法保证「零阻塞」。
  - 子问题 3（违反「返回值与副作用不并存」）：`CODE_PINCEPLES/CODE_PRICEPLES` 要求一个函数要么返回值、要么产生副作用。`TTSRequestSampler.should_sample` 在返回判定时自增 `_sampled`（`services/tts_request_sampler.py:42-44`），`record()` 追加后返回副本（`:58-60`），`TTSAlertService.check()` 写列表后返回命中项（`services/tts_alerts.py:35-41`），`DailyEvaluationQuota.reserve()` 修改计数后返回布尔（`services/tts_quality_quota.py:24-35`）。需拆成「查询/记录」两个动作，或显式返回含状态的记录对象。
  - 已决议 A：抽样率默认 5%，由 `TTS_QUALITY_SAMPLE_RATE` 可配置；按 `speaker_id`/`user_id` 分层的系统抽样（每个 stratum 每天约 5% 份额）与失败请求 100% 采样（`TTS_QUALITY_FAILURE_SAMPLE_RATE`，默认 1.0）。
  - 已决议 B：全量三项指标；参考文本取请求文本，参考音频取 `/tts` 可选 `reference_audio`；无参考时 similarity 为 `None` 且不参与判定。
  - 已决议 C：异步后台线程池；队列满载丢弃并记 `tts_quality_evaluation_dropped`。
  - 已决议 D：进程内写回统计 / 告警 / 采样记录；MongoDB 持久化见下方遗留项。
  - 已决议 E：并发与队列上限已生效；`TTS_QUALITY_DAILY_LIMIT`（默认 1000/日）超额只记录轻量结果并记 `tts_quality_evaluation_quota_exceeded`，不阻断响应。
  - [ ] 遗留：结果持久化（MongoDB 时序集合）、保留窗口与文本脱敏
    - 已具备：`infrastructure/tts/quality_record_repository.py::MongoTTSQualityRecordRepository` 懒加载 `pymongo`，以 `recorded_at` 建 time-series 集合并设 `expireAfterSeconds = TTS_QUALITY_RETENTION_DAYS × 86400`（默认 30 天）；`ALLOWED_FIELDS` 白名单只保留指标字段，`text` / `stratum` / 原始音频不落库；观察器异步 `save()`，写库失败只记 `tts_quality_record_persist_failed`。
    - 子问题 A（保留窗口只在建集合时生效）：`quality_record_repository.py:67-73` 仅在 `create_collection` 时设置 TTL；运行期修改 `TTS_QUALITY_RETENTION_DAYS` 不会作用于既有集合，配置项与实际行为不一致。需在启动时用 `collMod` 对账，或在文档中明确「变更需手工迁移」。
    - 子问题 B（建连接/建集合非线程安全且异常被吞）：`_get_collection`（`:56-65`）无锁，两个 worker 线程可能各建一个 `MongoClient`；`_ensure_timeseries`（`:72-78`）把创建失败一律按 debug 记录，权限或配置错误在生产日志里不可见。需单飞初始化，并至少以 warning 记录非「集合已存在」的失败。
  - [ ] 遗留：分层抽样、失败请求 100% 采样、每日评估次数上限与超预算降级
    - 已具备：`TTSRequestSampler.should_sample(result)` 按 `stratum` 做系统抽样（UTC 日切计数）；失败请求（非 pass/skipped、时长异常、重复片段）走 `TTS_QUALITY_FAILURE_SAMPLE_RATE`；`DailyEvaluationQuota` 线程安全、UTC 日切重置、限额 0 表示不限；超额只写轻量记录并记日志。
    - 子问题（stratum 计数无界）：`TTSRequestSampler._strata`（`services/tts_request_sampler.py:26,55`）以「UTC 日期 + stratum」为键持续累积且从不清理，长期运行键数按天线性增长。需按日清理过期键，或改为固定窗口计数。
- [ ] 补齐生产评估模型的部署与端到端验证
  - 现状证据：仓库无 `models/dnsmos.onnx`；`ProductionTTSQualityEvaluator` 依赖的 Whisper / DNSMOS ONNX / ECAPA 模型文件，以及 `onnxruntime` / `librosa` / `speechbrain` 等运行时依赖，均未纳入部署流程（本机默认 Python 亦缺失）。当前 `tests/test_tts_quality_integration.py` 以可注入的 fake evaluator 覆盖契约与装配，未跑真实模型推理；生产上模型缺失时观察器降级为日志 + 告警并放行响应。
  - 待人为抉择 A（模型来源与放置）：随镜像打包 / 启动时下载 / 挂载卷；模型版本、校验与回滚方式。
  - 待人为抉择 B（依赖与资源）：把推理依赖加入 `pyproject.toml` 的 `tts` extra 与镜像；CPU/GPU 选择、显存与并发上限。
  - 待人为抉择 C（验收环境与降级口径）：真实模型由谁提供、在哪个环境跑通 `/tts`；三项指标缺项是否算通过；部署文档需记录的模型路径、版本校验方式，以及「模型缺失 / 推理失败放行并告警」是否作为正式降级契约。
  - 人工确认前不实施：模型文件、推理依赖、镜像与部署文档改动都不在本轮范围内；fake evaluator 只覆盖契约与装配，不能据此勾选本项。
  - 验收：用真实 `sig_bak_ovr.onnx` 等模型跑通一次完整 `/tts` 抽样评估（WER / DNSMOS / Similarity 三项均有值），并在部署文档中记录模型路径与降级行为。
- [ ] 每日统计、滚动均值、P90/P95、异常比例（进程内按日聚合）
  - 已具备（2026-11-10 复核）：`TTSQualityStatistics` 有 UTC 日切分桶（可注入 `clock`）、`daily_statistics()` / `daily_breakdown()` / `anomaly_rate`（status 非 pass/skipped 或时长异常/重复片段计异常，`skipped` 不算）、`first_chunk_statistics()`；`TTSService.metrics()` 暴露 `daily` 与 `first_chunk`；`tests/test_tts_quality_statistics.py` 7 项单元测试绿灯。
  - 子问题 1（跨线程迭代崩溃，已复现）：`daily_statistics()`（`services/tts_quality_statistics.py:52`）用 `zip(self._results, self._days)`、`daily_breakdown()`（`:67`）用集合推导直接迭代 deque；观察器 worker 线程执行 `add()` 时，`/tts/metrics` 会抛 `RuntimeError: deque mutated during iteration`（并发探针可稳定复现）。需在锁内做快照，或让 `add` 与读接口统一用锁保护。
  - 子问题 2（缺测被当成 0）：`rolling_mean`（`:45-48`）用 `item.get(key, 0.0) or 0.0`，未评估记录会被计为 0 拉低均值，与 `_summary` 跳过 `None` 的口径不一致；当前无生产调用点，启用前需统一口径。
  - 子问题 3（无并发测试）：`tests/test_tts_quality_statistics.py` 只有单线程用例，未覆盖子问题 1 的竞态。
- [ ] 固定测试句集 + 每日回归测试 + 版本对比曲线
  - 现状证据：固定句集已有 3 句英文（`services/tts_regression_suite.py:11-15`），但只有手动接口 `POST /tts/regression`（`controllers/tts_controller.py:147-150`，限流 2/min），无任何每日调度（`main.py:201-212` 只初始化 `TaskScheduler` / MOLT 调度器并装配 TTS 服务；`services/task_scheduler.py:58-73` 仅 `register_root_task()`，未注册 TTS 回归任务）；`TTSService.run_regression` 只返回 `duration_seconds`/`bytes`（`services/tts_service.py:101-114`），未调用 `TTSRegressionSuite.run()`，`_runs` 历史不写入；`compare()`（`services/tts_regression_suite.py:40`）唯一调用点是 `tests/test_tts_regression_suite.py:18`；`version` 为随机 `uuid.uuid4()[:8]`（`services/tts_regression_suite.py:31,36`），无真实版本与持久化，无法形成曲线。
  - 待人为抉择 A（句集内容）：是否保留现有 3 句英文，改为中文/中英混合、覆盖多情绪与长难句、数字与专有名词；句集数量与存放位置（代码常量 / 配置文件 / 数据库）。
  - 待人为抉择 B（调度方式）：复用 `AsyncTaskScheduler` / `TaskScheduler` / 独立 cron / CI 定时任务；执行时间、时区、并发与失败重试策略。
  - 待人为抉择 C（回归指标）：在时长/字节之外是否加入 WER/DNSMOS/Similarity、首包延迟、失败率；阈值与判定为回归的标准。
  - 待人为抉择 D（版本标识）：用 checkpoint 名 / git sha / 模型+配置快照 替换随机 uuid；版本元数据从哪读取。
  - 待人为抉择 E（持久化与曲线）：结果存 MongoDB / 时序库 / 静态报表，对比基线（上一版本 vs 固定基线）、保留窗口、曲线展示方式与回归告警阈值。
  - 待人为抉择 F（手动接口）：是否保留 `POST /tts/regression` 及其 2/min 限流，还是改为只由调度触发。
  - 人工确认前不实施：A–F 未拍板前不新增调度、不改句集、不改 version 口径；可代码化的部分（`run_regression` 接 `TTSRegressionSuite.run()`、每日调度底座、真实版本与 `_runs` 持久化）同批暂停，避免按推测写死后返工。
  - 验收：A–F 确认后，固定句集能按日自动执行、`_runs` 历史可查、`compare()` 产出跨版本对比结果，并有可复现的测试证据链；不得仅凭「代码已装配」勾选。

## 6. 代码原则审计整改（2026-10-30）

> 说明：本节按「勾选真实性复核 / 失败测试 / 代码原则 / 质量门禁」分类。2026-11-05 全量核查后，无缺陷的勾选项已从本文件移除，仍有缺陷的项已回退为 `- [ ]` 并挂子任务。

### 6.1 TODO 勾选真实性复核

- [ ] 复核 §5 生产监控/统计/报警/反馈/回归
  - 进展：`TTSRequestSampler` / `TTSQualityStatistics` / `TTSAlertService` / `TTSAudioQualityService` 现已由 `bootstrap/tts_composition.py` 装配，并由 `/tts` 请求真实写入；通知实现 `StructuredLogNotifier` 已接入。
  - [ ] 子任务（`§6.1-7` / PARTIAL）：`TTSFeedbackMonitor` 仍只由手动反馈接口驱动，`TTSRegressionSuite` 仍无每日调度与版本曲线；每日聚合（§5-1）与评估记录持久化（§5 首条遗留）已于 2026-11-09 补齐基础实现，2026-11-10 复核后又挂出并发与保留窗口子问题。需逐项核实反馈 / 回归部分并补可复现证据链。
  - 待人为抉择 A（反馈数据来源）：`TTSFeedbackMonitor` 是否继续只由手动 `POST /tts/feedback`（`controllers/tts_controller.py:135-139`，限流 30/min）驱动，还是接入质量评估结果 / 客户端埋点自动写入；若新增自动通道，需定脱敏口径与写入频率。
  - 待人为抉择 B（回归证据口径）：每日回归与版本曲线的实现范围以 §5「固定测试句集 + 每日回归测试 + 版本对比曲线」A–F 的结论为准；确认前不实施、不勾选。
  - 人工确认前不实施：本项只做勾选真实性复核，反馈 / 回归两项待证据齐备后逐项重新核实；「服务已装配、请求有写入」不等于「监控链路已验证」，不得据此勾选。

### 6.4 质量门禁（防止回归）

- [ ] 复核后同步更新第 1–5 节勾选状态
  - [ ] 子任务（`§6.4-5` / PARTIAL）：核查发现多处原记录与代码不符，历史整改记录已移除；各回退项需补可复现证据后再重新勾选。
  - 待人为抉择 A（勾选口径）：是否维持「可复现测试证据 + 代码现状一致」的现行门槛，还是允许仅凭人工复核结论勾选；若放宽，需同步修改 `scripts/todo_guard.py` 门禁规则并说明理由（同类放宽门禁的提议此前已被否决，见 git 历史 `22d6233`）。
  - 待人为抉择 B（范围与顺序）：第 1–5 节回退项的复核顺序、由谁确认证据、是否要求一次全量回归；§5 模型部署与每日回归两项在各自 A–F 未确认前不得勾选。
  - 人工确认前不实施：保持 `[ ]`，不在无新证据的情况下批量同步勾选状态。
