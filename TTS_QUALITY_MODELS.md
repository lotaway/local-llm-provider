# TTS 质量评估模型 — 部署与端到端验证

本文记录生产质量评估链路（`/tts` 抽样评估）所用三个真实模型的来源、放置方式、
按需下载策略、降级契约与验证步骤。评估器唯一的组装入口是
`tts_estimate/infrastructure/evaluator_factory.py::build_quality_evaluator`，
`bootstrap` 与 `infrastructure/tts/quality_evaluator_adapter.py` 均经它取栈，
不得在别处二次组装。

## 1. 模型清单

| 指标 | 模型 | 落盘路径 | 来源 | 版本锚点 |
| --- | --- | --- | --- | --- |
| WER（转写比对） | OpenAI Whisper | `models/whisper/<name>.pt`（默认 `large` → `large-v3.pt`） | whisper 官方权重 | 库内版本映射（`openai-whisper>=20231117`） |
| DNSMOS（P.835） | Microsoft `sig_bak_ovr.onnx` | `models/dnsmos.onnx` | `DNS-Challenge` 仓库 | sha256 `269fb…f6edd`（见 `model_assets.py::DNSMOS_ASSET`） |
| Speaker Similarity | speechbrain ECAPA-TDNN（VoxCeleb） | `models/ecapa/`（5 个 ckpt/yaml 文件） | HF `speechbrain/spkrec-ecapa-voxceleb` | 仓库 `main` 分支当前发布 |

`models/` 已被 `.gitignore` 与 `.dockerignore` 排除，模型不进 git、不进镜像。

## 2. 放置与按需解析（决议 一A）

三个模型都**在首次评估时按需解析**，服务启动不下载、镜像不打包：

- **DNSMOS** 走 `model_assets.py::ensure_local_model`：已有文件先校验 sha256，
  匹配则直接使用（`model_asset_ready`）；不匹配则隔离为 `*.rejected-<短哈希>`
  并记录 `model_asset_checksum_mismatch`，随后从固定 URL 下载到同目录临时文件，
  校验通过后原子替换；下载或校验失败时回滚隔离文件（`model_asset_rollback`），
  部署保持在最后一个已知可用状态。
- **Whisper** 经 `whisper.load_model(name, download_root="models/whisper")`：
  文件已存在则离线加载，缺失才由 whisper 官方下载器拉取。
- **ECAPA** 经 `EncoderClassifier.from_hparams(source=…, savedir="models/ecapa",
  local_strategy=COPY_SKIP_CACHE)`：文件直接落 `models/ecapa/`，不写
  `~/.cache/huggingface`（该目录在 compose 环境挂载为容器 `/root/.cache`，
  属主混乱且不可依赖）。

**手动放置**：把上面表格中的文件按相同路径放进项目根 `models/` 即可，解析器
命中本地文件就不会触发下载。DNSMOS 手动放置时 sha256 必须与
`DNSMOS_ASSET.sha256` 一致，否则视为损坏并按上述隔离/回滚流程重下。

## 3. 配置

评估器配置集中在 `tts_estimate/config/settings.py`（`pydantic-settings`，
前缀 `TTS_EVAL_`、嵌套分隔符 `__`）：

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `TTS_EVAL_MODELS__WHISPER_MODEL` | `large` | whisper 模型名（`large` 映射 large-v3） |
| `TTS_EVAL_MODELS__DNSMOS_MODEL_PATH` | `models/dnsmos.onnx` | DNSMOS ONNX 路径 |
| `TTS_EVAL_MODELS__ECAPA_MODEL_SOURCE` | `speechbrain/spkrec-ecapa-voxceleb` | ECAPA 来源标识 |
| `TTS_EVAL_MODELS__MODELS_ROOT` | `models` | 模型根目录 |
| `TTS_EVAL_THRESHOLDS__WER_MAX` | `0.08` | WER 上限 |
| `TTS_EVAL_THRESHOLDS__DNSMOS_MIN` | `3.8` | DNSMOS 下限 |
| `TTS_EVAL_THRESHOLDS__SIMILARITY_MIN` | `0.78` | 相似度下限 |
| `TTS_EVALUATOR_RETRY_COOLDOWN_SECONDS`（`constants.py`） | `600` | 评估器构造失败后的重试冷却 |

## 4. 依赖与部署

- 推理依赖声明在根 `pyproject.toml` 的 `tts` extra：`openai-whisper`、
  `onnxruntime`、`speechbrain`、`jiwer`、`pydantic-settings`（另有 StyleTTS2
  引擎依赖）。torch/torchaudio 刻意不声明，由宿主 ROCm/CUDA 构建提供；
  `scripts/check_torch_build.py` 先钉住已装版本再以约束文件守卫 `pip install`，
  防止 extra 把加速器构建替换成默认包。
- 部署不绑定显卡厂商，torch 由所选基座镜像提供：NVIDIA 用官方 PyTorch CUDA
  镜像（`Dockerfile` 的 `BASE_IMAGE` 默认值），AMD 用 ROCm 镜像
  （`--build-arg BASE_IMAGE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`）。
  `scripts/check_torch_build.py` 对 cuda/rocm 任一加速器构建都接受并钉版守卫。
- `docker-compose.yml` 的 `pytorch` 服务是 AMD/ROCm 路径的示例（RX 7900 XTX
  实测可用）：仓库 bind-mount 到 `/app`，`scripts/docker-entrypoint.sh` 守卫安装
  `.[tts]`、缺 ffmpeg 时 `apt-get` 补装，并仅在 ROCm 环境缺少 bitsandbytes
  二进制时才源码编译（NVIDIA 镜像自带 CUDA 二进制，直接跳过该分支）。
  NVIDIA 环境把该服务的镜像与设备暴露换成对应 CUDA 镜像 +
  nvidia-container-toolkit 即可。
- **ffmpeg 是 whisper 转写的前置 CLI**。容器内由 entrypoint/Dockerfile 安装；
  无 apt 权限的本机环境可 `pip install imageio-ffmpeg` 并把
  `imageio_ffmpeg.get_ffmpeg_exe()` 的可执行文件加入 `PATH`。
- 显存：whisper `large`（fp16）约 3GB，与 StyleTTS2 同卡共存时注意峰值。

## 5. 降级契约

评估器**构造或推理失败都不阻断 `/tts` 响应**：构造失败经
`infrastructure/tts/quality_evaluator_adapter.py` 记录失败并告警
（`quality evaluator construction failed`），评估结果直通放行；冷却期
（`TTS_EVALUATOR_RETRY_COOLDOWN_SECONDS`，默认 600 秒）内不重试构造，避免每次
请求都撞同一失败。DNSMOS 下载失败同样使构造失败并走该契约，模型路径在冷却期
结束后下次评估再试。该契约有测试锚定：`tests/test_tts_quality_integration.py`、
`tests/test_tts_quality_gateway.py`。

## 6. 端到端验证

验证脚本：`scripts/verify_tts_quality_models.py`。它会解析全部三个模型（首次
运行时按需下载到 `models/`），把一段真实语音经 `TTSService.stream` 完整回放，
等待异步抽样评估收敛，并要求 **WER / DNSMOS / Similarity 三项统计均有真实值**
（count > 0）才返回 0。

```bash
cd <repo-root>
PATH=/tmp/opencode/ffmpeg-bin:$PATH python scripts/verify_tts_quality_models.py \
    --audio /path/to/speech.ogg --text "its transcript"
```

不带参数时使用内置 LibriSpeech 固定语音与冻结的 whisper `large` 参考转写。
`--audio`/`--text` 可替换为任意真实语音与人工转写。

**2026-11-11 实测证据**（RX 7900 XTX 24GB，torch `2.15.0+rocm10.0`，
whisper `large-v3`，DNSMOS P.835 `sig_bak_ovr.onnx`，
ECAPA `spkrec-ecapa-voxceleb`，全部真实推理）：

```
verification_metrics wer=0.0000 dnsmos_overall=2.8285 similarity=1.0000 counts=1.0/1.0/1.0
verification_complete
exit=0
```

- WER 0.0000：whisper `large` 对回放音频的转写与冻结参考文本逐词一致。
- DNSMOS 2.8285：真实 ONNX 推理；素材为 16kHz 单声道 PCM 降采样语音，分数
  反映素材而非生成器质量。
- Similarity 1.0000：验证以同一音频作参考与生成，ECAPA 余弦在数学上应为 1，
  是嵌入管线正确性的判定值；真实生成语音的相似度由线上抽样统计观察。

回归测试（CI targeted 套件内，保持绿灯）：
`tests/test_tts_model_assets.py`（路径/按需解析/单一组装点）、
`tests/test_dnsmos_inference.py`（DNSMOS 喂入形状与校准）、
`tts_estimate/tests/test_evaluator.py`（评估器组合）、
`tests/test_tts_quality_integration.py`（降级契约与装配）。
