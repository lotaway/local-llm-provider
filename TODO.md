# TODO — 未完成任务列表

> 整理于 2026-11-05：依据 `audit-2026-11-05.md` 移除 58 项已核验完成的任务；其余 31 项 PARTIAL + 1 项 NO_EVIDENCE 回退为待办并挂子任务，另新增 3 项 P0 运行期缺陷。
> 约定（2026-11-10 起）：本文件只保留未完成任务；逐项核验通过的任务直接从本文件移除，证据留在 git 历史与对应测试中。若再出现勾选项，其定向测试须在 `scripts/todo_test_map.json` 与 `.github/workflows/ci.yml` 中登记并保持绿灯。
> 2026-11-07：按「离线筛选 / 抽样告警、异步评估、不阻断响应」决议落地后，§4 首条、§5 首条与 §6.1-5 曾改为 `[x]`。
> 2026-11-09：按「TTS 质量优先、不设 500ms 绝对阈值」决议改写 §4-7；§5-1 每日聚合与 §6.4 门禁红项曾勾选。同批补齐首包耗时观测、分层/失败抽样、每日配额与评估记录持久化。
> 2026-11-10（人工确认）：未完成项均需人工确认后才能实施，人工确认点逐项写在对应任务下：§5 模型部署 A–C、§5 每日回归 A–F、§6.1-7 A–B、§6.4-5 A–B。
> 2026-11-10（完成项复核）：逐条核对 10 个 `[x]`（6 项主任务 + 4 项子任务）。核验通过并移除 5 个：§4-7 流式首包观测（主+子）、§6.1 §4 网关接入复核（主+子）、§6.4 AST 门禁红项。回退为 `[ ]` 并挂子问题 5 个：§4 质量链路接入、§5 抽样监控（主+两条遗留）、§5-1 每日统计，缺陷与复现方式见各条。
> 2026-11-10（缺陷修复）：修复复核挂出的 11 个子问题后逐项核验通过，移除 §4 质量链路接入、§5 抽样监控（含两条遗留）、§5-1 每日统计。时长改按 WAV 头（PCM_16）解析，实测 3.0s 不再算成 1.5s；统计读写统一快照加锁，并发探针不再抛 `RuntimeError: deque mutated during iteration`，`rolling_mean` 跳过缺测；去重改用有界摘要集合且 `_record_request` 移交线程池；告警/采样记录设上限并脱敏，一次请求只告警一次；保留窗口在集合已存在时用 `collMod` 对账，建连接单飞初始化，非「已存在」失败以 warning 记录。证据：`tests/test_tts_audio_quality.py`、`tests/test_tts_quality_statistics.py`、`tests/test_tts_request_sampler.py`、`tests/test_tts_alerts.py`、`tests/test_tts_quality_integration.py`（101 项映射测试绿灯）。

## 5. TTS 质量评估与线上监控（来源：TTS_QUALITY_ESTIMATE.md）

- [ ] 补齐生产评估模型的部署与端到端验证
  - 现状证据：仓库无 `models/dnsmos.onnx`；`ProductionTTSQualityEvaluator` 依赖的 Whisper / DNSMOS ONNX / ECAPA 模型文件，以及 `onnxruntime` / `librosa` / `speechbrain` 等运行时依赖，均未纳入部署流程（本机默认 Python 亦缺失）。当前 `tests/test_tts_quality_integration.py` 以可注入的 fake evaluator 覆盖契约与装配，未跑真实模型推理；生产上模型缺失时观察器降级为日志 + 告警并放行响应。
  - 待人为抉择 A（模型来源与放置）：随镜像打包 / 启动时下载 / 挂载卷；模型版本、校验与回滚方式。
  - 待人为抉择 B（依赖与资源）：把推理依赖加入 `pyproject.toml` 的 `tts` extra 与镜像；CPU/GPU 选择、显存与并发上限。
  - 待人为抉择 C（验收环境与降级口径）：真实模型由谁提供、在哪个环境跑通 `/tts`；三项指标缺项是否算通过；部署文档需记录的模型路径、版本校验方式，以及「模型缺失 / 推理失败放行并告警」是否作为正式降级契约。
  - 人工确认前不实施：模型文件、推理依赖、镜像与部署文档改动都不在本轮范围内；fake evaluator 只覆盖契约与装配，不能据此勾选本项。
  - 验收：用真实 `sig_bak_ovr.onnx` 等模型跑通一次完整 `/tts` 抽样评估（WER / DNSMOS / Similarity 三项均有值），并在部署文档中记录模型路径与降级行为。
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
  - 进展：`TTSRequestSampler` / `TTSQualityStatistics` / `TTSAlertService` / `TTSAudioQualityService` 现已由 `bootstrap/tts_composition.py` 装配，并由 `/tts` 请求真实写入；通知实现 `StructuredLogNotifier` 已接入。抽样、统计、告警与持久化的缺陷已于 2026-11-10 修复并复核通过、移出本文件。
  - [ ] 子任务（`§6.1-7` / PARTIAL）：`TTSFeedbackMonitor` 仍只由手动反馈接口驱动，`TTSRegressionSuite` 仍无每日调度与版本曲线。需逐项核实反馈 / 回归部分并补可复现证据链。
  - 待人为抉择 A（反馈数据来源）：`TTSFeedbackMonitor` 是否继续只由手动 `POST /tts/feedback`（`controllers/tts_controller.py:135-139`，限流 30/min）驱动，还是接入质量评估结果 / 客户端埋点自动写入；若新增自动通道，需定脱敏口径与写入频率。
  - 待人为抉择 B（回归证据口径）：每日回归与版本曲线的实现范围以 §5「固定测试句集 + 每日回归测试 + 版本对比曲线」A–F 的结论为准；确认前不实施、不勾选。
  - 人工确认前不实施：本项只做勾选真实性复核，反馈 / 回归两项待证据齐备后逐项重新核实；「服务已装配、请求有写入」不等于「监控链路已验证」，不得据此勾选。

### 6.4 质量门禁（防止回归）

- [ ] 复核后同步更新第 1–5 节勾选状态
  - [ ] 子任务（`§6.4-5` / PARTIAL）：2026-11-10 按「可复现证据 + 代码现状一致」再次复核，已移除 3 项并修复其挂出的 11 个子问题；剩余 §5 两项与 §6.1-7 待人工确认后再复核。
  - 待人为抉择 A（勾选口径）：是否维持「可复现测试证据 + 代码现状一致」的现行门槛，还是允许仅凭人工复核结论勾选；若放宽，需同步修改 `scripts/todo_guard.py` 门禁规则并说明理由（同类放宽门禁的提议此前已被否决，见 git 历史 `22d6233`）。
  - 待人为抉择 B（范围与顺序）：第 1–5 节回退项的复核顺序、由谁确认证据、是否要求一次全量回归；§5 模型部署与每日回归两项在各自 A–F 未确认前不得勾选。
  - 人工确认前不实施：保持 `[ ]`，不在无新证据的情况下批量同步勾选状态。
