# TTS Estimate - 独立项目说明

本项目用于生产级 TTS (Text-to-Speech) 生成质量的自动化评估。

## 1. 核心配置方案

项目使用 `pydantic-settings` 管理配置。你可以通过环境变量或项目根目录下的 `.env` 文件进行配置（具体参考 `.env.example`）。

### 环境变量说明
- `TTS_EVAL_THRESHOLDS__WER_MAX`: WER 的上限（如 0.08 表示 8%）。
- `TTS_EVAL_THRESHOLDS__DNSMOS_MIN`: DNSMOS 综合分数的底线（推荐 3.8-4.0）。
- `TTS_EVAL_THRESHOLDS__SIMILARITY_MIN`: 说话人相似度的底线（推荐 0.78-0.82）。
- `TTS_EVAL_MODELS__DNSMOS_MODEL_PATH`: 官方 `sig_bak_ovr.onnx` 的本地路径。

## 2. 模型下载指引

为了让项目正常运行，你需要准备以下模型文件：

### A. DNSMOS (ONNX)
- **来源**: [Microsoft DNS-Challenge 仓库](https://github.com/microsoft/DNS-Challenge) 的 `DNSMOS/DNSMOS/sig_bak_ovr.onnx`（P.835 三分模型）。
- **操作**: 下载后放置到 `models/dnsmos.onnx`。
- **不要使用** `DNSMOS/model_v8.onnx`：那是 P.808 单分模型，输入为 120 维 mel 谱 `[1, T, 120]`，与波形输入契约不兼容。

### A.1 推理契约（对齐官方 `dnsmos_local.py`）

| 项目 | 取值 |
| --- | --- |
| 采样率 | 16 kHz 单声道 float32 |
| 单帧长度 | 9.01 秒 = 144160 采样 |
| 输入张量 | `[1, 144160]`，输入名取自 `session.get_inputs()[0].name`（官方为 `input_1`） |
| 窗口步长 | 1 秒（16000 采样），长音频多窗取算术平均 |
| 短音频 | 重复拼接补齐至 ≥144160 采样 |
| 输出 | 首个输出 `[1, 3]`，依次 `SIG_raw / BAK_raw / OVR_raw` |
| 校准 | 官方 polyfit：SIG `[-0.08397278, 1.22083953, 0.0052439]`、BAK `[-0.13166888, 1.60915514, -0.39604546]`、OVR `[-0.06766283, 1.11546468, 0.04602535]` |
| 领域映射 | `overall=OVR`、`signal=SIG`、`background=BAK` |

构造 `MicrosoftDNSMOSCalculator(..., apply_calibration=False)` 可拿到未校准的 raw 分数，用于与官方 CSV 的 `SIG_raw / BAK_raw / OVR_raw` 列对照；`session=` 可注入 ONNX Runtime session，便于离线单测。

### B. Whisper
- **操作**: 首次运行程序时，`openai-whisper` 会自动从 HuggingFace 下载指定模型（默认 `large`）。如果要在无网络环境运行，请提前缓存模型至 `~/.cache/whisper`。

### C. ECAPA-TDNN
- **操作**: `speechbrain` 会在首次运行时下载模型。可以通过配置 `TTS_EVAL_MODELS__ECAPA_MODEL_SOURCE` 指定本地路径。
