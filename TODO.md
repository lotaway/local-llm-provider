# TODO — 未完成任务列表

> 整理于 2026-11-05：依据 `audit-2026-11-05.md` 移除 58 项已核验完成的任务；其余 31 项 PARTIAL + 1 项 NO_EVIDENCE 回退为待办并挂子任务，另新增 3 项 P0 运行期缺陷。
> 本文件以未完成任务为主体，已核验完成的项保留 `[x]` 与证据；已移除内容与历史整改记录见 git 历史 `44d1d81`。
> 2026-11-07：按「离线筛选 / 抽样告警、异步评估、不阻断响应」决议落地后，§4 首条、§5 首条与 §6.1-5 改为 `[x]` 并保留完成证据；勾选项的定向测试须在 `scripts/todo_test_map.json` 与 `.github/workflows/ci.yml` 中登记且保持绿灯。
> 2026-11-09：按「TTS 质量优先、不设 500ms 绝对阈值」决议改写 §4-7；§5-1 每日聚合与 §6.4 门禁红项完成并勾选。

## 4. TTS 情感语音系统（来源：TTS_SPEC.md）

- [x] 将质量评估流水线接入 TTS 生成链路（2026-11-07 决议：离线筛选 / 抽样告警，不在线阻断响应）— 证据 `tests/test_tts_quality_integration.py`
  - 完成证据：`bootstrap/tts_composition.py` 已装配 `ProductionTTSQualityEvaluator` → `TTSQualityGateway` → `TTSQualityObserver` → `TTSService.quality_observer`；`ITTSQualityEvaluator` 有生产实现（bytes → 临时 WAV、懒加载 Whisper/DNSMOS/ECAPA）；`/tts` 支持可选 `reference_audio`；`_to_report` 改从 `metrics.speaker_similarity` 取数；evaluator 不可用返回 `skipped` 并放行；`_record_request` 修正递交顺序后 WER/DNSMOS/Similarity 能进入 `metrics()`。流式多段 WAV 由 `merge_wav_segments` 合并为合法单 WAV 后再送模型。17 项契约/装配测试绿灯。
  - 原条目「自动拦截」语义经决议不实施：在线阻断、HTTP 错误体、异步重生成不再作为本项验收条件；响应已发出后抛 `TTSQualityRejected` 的死代码已移除。
  - 已决议 A：仅事后评估 + 告警，不阻断；「首包 <500ms」是 STT 类硬实时推导，TTS 全量合成后再评估与之无冲突。
  - 已决议 B：请求可选携带 `reference_audio`（base64），无参考时跳过 ECAPA 与该阈值，降级为 WER + DNSMOS。
  - 已决议 C：复用 `tts_estimate` 三项模型并按需懒加载，仅对抽样请求推理。
  - 已决议 D：阈值沿用 `constants.py` 的 0.08 / 3.8 / 0.78；evaluator 不可用、推理失败、队列满载均放行并告警。
  - 已决议 E：不阻断响应，无 HTTP 错误语义；失败只写一条 `evaluation_failed` 告警。
- [ ] 实现流式输出（质量优先；首包延迟仅作观测指标，不设 500ms 绝对阈值）
  - [ ] 子任务（`§4-7` / PARTIAL）：2026-11-09 决议——这是 **TTS**（`todo/TTS_SPEC.md` §4.4），不是 STT 式实时识别，不为压低首包延迟牺牲质量：不切碎音频、不削减 diffusion 步数、不加 SLA 断言。`model_providers/tts/style_tts2_adapter.py:106-117` 的 `first_chunk/elapsed_ms` 仅用于观测，`StyleTTS2Engine.synthesize_stream` 按句产出完整 WAV，质量评估在合成后异步进行且不阻断响应（`tests/test_tts_quality_integration.py`）。
  - 剩余缺口：`TTSService.stream`（`services/tts_service.py:62-64`）调用 `generate_speech_stream` 时**未传 `callback`**，首包耗时事件在生产链路里无人消费。需把该观测值写进请求记录 / 趋势，仍不设阈值、不告警、不拦截；验收为能在 `metrics()` 查到首包耗时观测序列。
## 5. TTS 质量评估与线上监控（来源：TTS_QUALITY_ESTIMATE.md）

- [x] 生产监控抽样（默认 5%）接入评估流水线 — 证据 `tests/test_tts_quality_integration.py`
  - 完成证据：`TTSQualityObserver` 由 `TTSService._record_request` 递交，抽样命中后经有界线程池（`TTS_QUALITY_MAX_WORKERS` / `TTS_QUALITY_MAX_PENDING`，默认 2 / 8）异步跑 `ProductionTTSQualityEvaluator`，完成后集中写回 `TTSQualityStatistics` / `TTSRequestSampler` / `TTSAlertService`；WER/DNSMOS/Similarity 不再恒为 0；请求路径零阻塞，模型缺失或推理失败只告警、不影响响应。
  - 已决议 A：抽样率默认 5%，由 `TTS_QUALITY_SAMPLE_RATE` 可配置；分层抽样与失败请求 100% 采样未纳入本期。
  - 已决议 B：全量三项指标；参考文本取请求文本，参考音频取 `/tts` 可选 `reference_audio`；无参考时 similarity 为 `None` 且不参与判定。
  - 已决议 C：改为异步后台线程池；队列满载丢弃并记 `tts_quality_evaluation_dropped`。
  - 已决议 D：进程内写回统计 / 告警 / 采样记录；保留窗口与文本脱敏未纳入本期。
  - 已决议 E：并发与队列上限已生效；模型缺失或推理失败放行并告警；每日次数上限未纳入本期。
  - [ ] 遗留：结果持久化（MongoDB 时序集合）、保留窗口与文本脱敏。
  - [ ] 遗留：分层抽样、失败请求 100% 采样、每日评估次数上限与超预算降级。
- [ ] 补齐生产评估模型的部署与端到端验证
  - 现状证据：仓库无 `models/dnsmos.onnx`；`ProductionTTSQualityEvaluator` 依赖的 Whisper / DNSMOS ONNX / ECAPA 模型文件，以及 `onnxruntime` / `librosa` / `speechbrain` 等运行时依赖，均未纳入部署流程（本机默认 Python 亦缺失）。当前 `tests/test_tts_quality_integration.py` 以可注入的 fake evaluator 覆盖契约与装配，未跑真实模型推理；生产上模型缺失时观察器降级为日志 + 告警并放行响应。
  - 待人为抉择 A（模型来源与放置）：随镜像打包 / 启动时下载 / 挂载卷；模型版本、校验与回滚方式。
  - 待人为抉择 B（依赖与资源）：把推理依赖加入 `pyproject.toml` 的 `tts` extra 与镜像；CPU/GPU 选择、显存与并发上限。
  - 验收：用真实 `sig_bak_ovr.onnx` 等模型跑通一次完整 `/tts` 抽样评估（WER / DNSMOS / Similarity 三项均有值），并在部署文档中记录模型路径与降级行为。
- [x] 每日统计、滚动均值、P90/P95、异常比例（进程内按日聚合）— 证据 `tests/test_tts_quality_statistics.py`
  - 完成证据：复核发现 P90/P95 / rolling_mean / failure_rate 原已实现（此前描述有误）；本次在 `TTSQualityStatistics` 新增 UTC 日切分桶（可注入 `clock`）、`daily_statistics()` 单日汇总、`daily_breakdown()` 近 N 日趋势、按日 `anomaly_rate`（status 非 pass/skipped 或时长异常 / 重复片段计为异常，`skipped` 不计异常），并在 `TTSService.metrics()` 暴露 `daily` 字段。6 项单元测试绿灯，与 §4 / §5 抽样链路合跑 43 passed。
  - 决议：本项验收范围限定为**进程内按日聚合与异常比例**；结果持久化、保留窗口与文本脱敏仍按 §5 首条下的遗留项保持 `[ ]`，未纳入本次勾选。
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

- [x] 复核 §4 质量网关接入 — 证据 `tests/test_tts_quality_integration.py`
  - 结论：`TTSQualityGateway` 现由 `bootstrap/tts_composition.py::_build_quality_observer` 在生产组合根实例化，并经 `TTSQualityObserver` 进入 `/tts` 链路；`ITTSQualityEvaluator` 已有生产实现 `infrastructure/tts/quality_evaluator_adapter.py::ProductionTTSQualityEvaluator`；`ITTSNotifier` 已由 `StructuredLogNotifier` 装配。原 `TTSService.quality_gateway` 端口改为 `quality_observer`，不再存在默认 `None` 的空挂。
  - [x] 子任务（`§6.1-5`）：已与 §4 首条合并完成，装配契约由 `tests/test_tts_quality_integration.py::test_composition_root_builds_a_service_with_the_observer` 覆盖。
- [ ] 复核 §5 生产监控/统计/报警/反馈/回归
  - 进展：`TTSRequestSampler` / `TTSQualityStatistics` / `TTSAlertService` / `TTSAudioQualityService` 现已由 `bootstrap/tts_composition.py` 装配，并由 `/tts` 请求真实写入；通知实现 `StructuredLogNotifier` 已接入。
  - [ ] 子任务（`§6.1-7` / PARTIAL）：`TTSFeedbackMonitor` 仍只由手动反馈接口驱动，`TTSRegressionSuite` 仍无每日调度与版本曲线；评估结果的持久化仍缺（每日聚合已于 2026-11-09 补上，见 §5-1）。需逐项核实并补可复现证据链。
### 6.4 质量门禁（防止回归）

- [ ] 复核后同步更新第 1–5 节勾选状态
  - [ ] 子任务（`§6.4-5` / PARTIAL）：核查发现多处原记录与代码不符，历史整改记录已移除；各回退项需补可复现证据后再重新勾选。
- [x] 修复质量门禁的既有红项：`tests/test_molt_startup.py` 的 AST 断言与 `main.py` 调用形态不符 — 证据 `tests/test_molt_startup.py`
  - 完成证据：根因是测试只收集 `ast.Attribute` 调用名，而 `main.py:95` 的 `build_molt_controller(current_memory_repo())` 是 `ast.Name` 调用。测试新增 `_call_names()` 同时收集 Name / Attribute，`main.py` 未改动；`python -m pytest tests/test_molt_startup.py -q` → **2 passed**（此前 1 failed / 1 passed），`todo_guard.py --report` 不再因该用例判违规。
  - 决议：待抉择 A 采用「改测试、不动生产代码」；待抉择 B（放宽 `failing_cases()` 只校验登记用例）**不实施**，避免门禁被架空。CI `Assert checked TODO tests are green` 恢复可用。
