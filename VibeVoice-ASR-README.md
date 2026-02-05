# VibeVoice‑ASR 使用教程（含本地视频字幕、会议录像翻译、实时系统音频字幕）

## 教程目标
- 统一讲清三种常见场景：本地视频生成时间轴字幕、会议录像转写并翻译、实时系统音频生成字幕
- 用官方推荐的两种方式部署：Docker（GPU）与从 GitHub 安装（本地）
- 输出结构化转录（说话人+时间戳+文本），必要时产出 SRT/VTT 字幕与双语字幕

## 能力概览
- 单次处理最长 60 分钟音频，保持全局语义一致与说话人跟踪
- 输出结构化结果：Who（说话人）、When（时间戳）、What（文本）
- 支持自定义热词、50+语言、多语言混说
- 提供 Gradio Demo、直接从文件推理脚本、加速推理的 vLLM 方案

## 环境准备
### GPU（推荐，Linux + NVIDIA）：使用 NVIDIA PyTorch 容器
拉起容器：
```bash
sudo docker run --privileged --net=host --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus all --rm -it nvcr.io/nvidia/pytorch:25.12-py3
```
如容器无 FlashAttention，可手动安装：
```bash
pip install flash-attn --no-build-isolation
```

### 本地安装（Mac/Windows/Linux）
安装与代码获取：
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```
安装 ffmpeg（用于视频音频提取与 Demo）：
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt update && sudo apt install ffmpeg -y`

Mac 注意事项（MPS/性能）：
- 大模型 7B 在纯 CPU/MPS 下较慢；尽量选短音频或使用离线批处理
- 建议设置环境变量以容错：`export PYTORCH_ENABLE_MPS_FALLBACK=1`
- 如遇显存/内存压力，降低并发、关闭加速组件或转用 Docker+GPU

## 快速体验
### Gradio Demo（可上传音频/视频并一键转录）
```bash
python demo/vibevoice_asr_gradio_demo.py --model_path microsoft/VibeVoice-ASR --share
```
浏览器打开后，上传文件开始转写；可在界面中查看说话人标注与时间戳。适合快速验证效果与热词配置。

## 本地视频生成时间轴字幕
目标：从视频提取音轨，生成含时间戳与说话人标注的字幕文件（SRT/VTT），可烧录回视频。

### 步骤 1：提取音频（示例：转 16k 单声道 wav）
```bash
ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 input.wav
```

### 步骤 2：运行文件推理脚本
```bash
python demo/vibevoice_asr_inference_from_file.py --model_path microsoft/VibeVoice-ASR --audio_files input.wav
```
输出会包含分段的说话人、文本与起止时间；也可导出结构化 JSON。

### 步骤 3：生成 SRT（示例将结构化 JSON 段落写成 SRT）
结构化数据通常类似：
```json
{
  "segments": [
    { "speaker": 0, "text": "...", "start": 0.0, "end": 8.2 },
    { "speaker": 1, "text": "...", "start": 8.3, "end": 15.0 }
  ]
}
```
将其转 SRT（示例 Python）：
```python
import json, datetime

def fmt(t):
    ms = int((t - int(t)) * 1000)
    return f"{str(datetime.timedelta(seconds=int(t))).zfill(8)},{ms:03d}"

data = json.load(open("asr_output.json"))
with open("output.srt", "w", encoding="utf-8") as f:
    for i, seg in enumerate(data["segments"], 1):
        f.write(f"{i}\n")
        f.write(f"{fmt(seg['start'])} --> {fmt(seg['end'])}\n")
        f.write(f"Speaker {seg['speaker']}: {seg['text']}\n\n")
```

### 步骤 4：将字幕烧录回视频（或外挂）
外挂字幕：
```bash
ffmpeg -i input.mp4 -i output.srt -c copy -c:s mov_text output_with_subs.mp4
```
直接烧录（硬字幕，无法关闭）：
```bash
ffmpeg -i input.mp4 -vf subtitles=output.srt -c:a copy output_burned.mp4
```

## 会议录像翻译与双语字幕
目标：在原文字幕基础上，生成目标语言译文并做双语显示（上行原文，下行译文）。

### 步骤 1：先按上节生成结构化转录/字幕（SRT 或 JSON）

### 步骤 2：调用翻译（开源方案示例，HuggingFace Transformers）
```bash
pip install transformers sentencepiece accelerate
```
以 M2M100 为例（英文→中文）：
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/m2m100_418M"  # 或 NLLB-200 模型
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def translate(text, src="en", tgt="zh"):
    inputs = tok(text, return_tensors="pt")
    gen = model.generate(**inputs, forced_bos_token_id=tok.get_lang_id(tgt))
    return tok.batch_decode(gen, skip_special_tokens=True)[0]
```

### 步骤 3：生成双语 SRT（每条字幕两行：原文+译文）
```python
import json, datetime

def fmt(t):
    ms = int((t - int(t)) * 1000)
    return f"{str(datetime.timedelta(seconds=int(t))).zfill(8)},{ms:03d}"

data = json.load(open("asr_output.json"))
with open("output_bilingual.srt", "w", encoding="utf-8") as f:
    for i, seg in enumerate(data["segments"], 1):
        src = seg["text"]
        tgt = translate(src, src="auto", tgt="zh")  # 自适应源语
        f.write(f"{i}\n")
        f.write(f"{fmt(seg['start'])} --> {fmt(seg['end'])}\n")
        f.write(src + "\n")
        f.write(tgt + "\n\n")
```
外挂/烧录双语字幕同上（ffmpeg）。

## 实时系统音频生成字幕
目标：在线看视频、线上会议时，将“系统声卡输出”回传为麦克风输入，实时转写为字幕。

### 方案 A（通用、简单）：虚拟声卡回传 + Gradio Demo
安装虚拟声卡：
- macOS：BlackHole（2ch/16ch）或 Loopback；安装后在“声音设置”中将“输出设备”与“输入设备”都指向虚拟声卡

打开 Gradio Demo：
```bash
python demo/vibevoice_asr_gradio_demo.py --model_path microsoft/VibeVoice-ASR
```
在浏览器界面选择“麦克风录音”，此时系统音频会被当作麦克风输入，实时生成字幕。若希望叠加到视频画面，可用 OBS：添加“浏览器来源”指向字幕网页，或用自定义前端展示。

### 方案 B（更灵活）：FFmpeg 拉系统音频管道 + 分块转写
macOS 用 avfoundation 捕获系统音频，推送到管道（示例）：
```bash
ffmpeg -f avfoundation -i ":0" -ac 1 -ar 16000 -f wav pipe:1 | tee realtime.wav > /dev/null
```
后台脚本每 N 秒切片并调用文件推理脚本，滚动输出字幕（可结合 TUI/网页叠加）。

实战建议：
- 在线会议需要最低延时：选短切片（3–5 秒），字幕刷新更频繁
- 直播/看视频可平衡质量与延时：切片 8–10 秒，错词更少
- 如需说话人分离，尽量保证语音干净并避免长时间多人同声重叠

## 热词与上下文提示
- 用于人名、术语、地名等提升准确率（Playground/Demo 通常提供输入框）
- 批处理脚本也可通过参数或文件传入定制上下文（如包含关键词列表）

## 常见问题与优化
- 处理很长的音频：模型支持 60 分钟单次输入；更长可分段处理并在后处理阶段拼接
- 内存不足/速度慢：优先使用 GPU；Mac MPS 退化时启用 `PYTORCH_ENABLE_MPS_FALLBACK=1`
- 语言混说与翻译：模型原生识别多语言与混说；翻译建议独立调用 MT 模型，生成双语字幕
- 字幕对齐与换行：每段控制在 5–12 秒，保证可读性；文本过长可按标点拆分再写入 SRT
- 导出格式：结构化 JSON 更适合二次处理；SRT/VTT 适合播放器外挂或烧录

## 参考与入口
- 文档：`docs/vibevoice-asr.md`（含 Demo 与文件推理用法）
- Demo：`demo/vibevoice_asr_gradio_demo.py`
- 文件推理：`demo/vibevoice_asr_inference_from_file.py`
- 模型主页与 Playground：仓库首页提供入口
- vLLM 加速：适合批处理与更快推理的场景

## 适用场景提示
- 在线会议：方案 A 最快落地；结合 OBS/字幕叠加用于直播或录屏
- 会议录像：先批量转写，结构化 JSON 存档；再生成中文/英文双语 SRT 便于复盘
- 看在线视频：虚拟声卡回传 + 短分片字幕，配合前端叠加效果最佳

