# TTS Estimate - 独立项目说明

本项目用于生产级 TTS (Text-to-Speech) 生成质量的自动化评估。

## 1. 核心配置方案

项目使用 `pydantic-settings` 管理配置。你可以通过环境变量或项目根目录下的 `.env` 文件进行配置（具体参考 `.env.example`）。

### 环境变量说明
- `TTS_EVAL_THRESHOLDS__WER_MAX`: WER 的上限（如 0.08 表示 8%）。
- `TTS_EVAL_THRESHOLDS__DNSMOS_MIN`: DNSMOS 综合分数的底线（推荐 3.8-4.0）。
- `TTS_EVAL_THRESHOLDS__SIMILARITY_MIN`: 说话人相似度的底线（推荐 0.78-0.82）。
- `TTS_EVAL_MODELS__DNSMOS_MODEL_PATH`: `dnsmos.onnx` 文件的本地路径。

## 2. 模型下载指引

为了让项目正常运行，你需要准备以下模型文件：

### A. DNSMOS (ONNX)
- **来源**: [Microsoft DNS-Challenge 仓库](https://github.com/microsoft/DNS-Challenge)的DNSMOS/DNSMOS/model_v8.onnx。
- **操作**: 下载官方的 `sig_bak_ovr.onnx` 或同类模型，并将其放置在 `models/dnsmos.onnx`。

### B. Whisper
- **操作**: 首次运行程序时，`openai-whisper` 会自动从 HuggingFace 下载指定模型（默认 `large`）。如果要在无网络环境运行，请提前缓存模型至 `~/.cache/whisper`。

### C. ECAPA-TDNN
- **操作**: `speechbrain` 会在首次运行时下载模型。可以通过配置 `TTS_EVAL_MODELS__ECAPA_MODEL_SOURCE` 指定本地路径。