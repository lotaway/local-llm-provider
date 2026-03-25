# TTS Emotion System SPEC (for Agent)

## 1. 目标（Goals）

构建一套可被 Agent 调用的高质量语音生成系统，具备以下能力：

1. 支持自然语音生成（接近真实人声）
2. 支持连续情绪表达（非离散标签）
3. 支持快速语音克隆（无需频繁训练）
4. 支持流式输出（低延迟播放）
5. 支持用户级语音与情绪偏好记忆
6. 可自动评估语音质量与情绪一致性
7. 可作为 LLM Provider 的标准能力被调用

---

## 2. 非目标（Non-Goals）

1. 不追求从零训练 TTS 模型
2. 不构建复杂 GUI 或编辑工具
3. 不实现音频后期专业制作（如混音工程级）
4. 不支持完全离线端侧推理（优先服务化）
5. 不追求极低资源设备运行（默认 GPU）

---

## 3. 系统整体架构

```text
LLM (文本 + 情绪推理)
↓
Emotion Parser（结构化情绪）
↓
TTS Engine (StyleTTS2 / Third-party Adapter)
↓
Post Processing (自然度增强 / Evaluation Filter)
↓
Storage & Memory (TTS Isolated Collections)
↓
Output (Stream / Audio File)
```

---

## 4. 核心功能点（Features）

### 4.1 文本转语音（基础能力）
- 输入：文本
- 输出：语音（wav/stream）
- 支持多语言（支持中/英/日）

---

### 4.2 情感建模（核心能力）
支持连续情绪参数 (VAD模型)：

```json
{
  "valence": -1.0 ~ 1.0,   // 愉悦度 (Negative to Positive)
  "arousal": 0.0 ~ 1.0,   // 激活度 (Calm to Excited)
  "dominance": 0.0 ~ 1.0, // 支配感 (Submissive to Dominant)
  "speed": 0.5 ~ 1.5,
  "pitch": -5 ~ +5,
  "energy": 0.0 ~ 1.0
}
```

---

### 4.3 语音克隆（低成本）
- 输入：参考音频（5s~2min）
- 输出：Speaker Embedding
- 检索逻辑：优先命中缓存，无则从存储加载

---

### 4.4 流式输出
- 延迟目标：首包延迟 < 500ms
- 通过 Chunk-based 生成实现

---

## 5. 关键技术方案 (Implementation Strategy)

### 5.1 引擎集成 (Engine Integration)
- **方案**：采用第三方模块化集成 (Vendor/Third-party)。
- **路径**：`third_party/StyleTTS2/`。
- **适配器**：`model_providers/tts/style_tts2_adapter.py` 负责隔离模型代码与主服务。

### 5.2 硬件兼容性 (Hardware Compatibility)
支持跨平台硬件加速，确保团队成员在不同设备上均可开发测试：
- **英伟达 (NVIDIA)** / **AMD**：通过 `torch.device("cuda")` (ROCm 在 PyTorch 中映射为 CUDA 接口)。
- **苹果 (Mac MPS)**：通过 `torch.backends.mps.is_available()` 动态切换。
- **CPU Fallback**：作为开发测试的最后兜底。
- **统一工具**：集中在 `utils/device.py` 中管理。

### 5.3 存储隔离 (Storage Isolation - Scheme B)
严格区分 TTS 数据与 Agent Brain (MOLT) 和 RAG 数据，采用严格 Prefix 模式：

#### **MongoDB (Metadata)**
Collection 命名强制使用 `tts_` 前缀，独立于 RAG 和 Brain：
- `tts_speakers`: 存储克隆出的模型元数据与 Embedding 路径。
- `tts_user_profiles`: 存储用户对声音的个性化偏好。

#### **Redis (Cache)**
全局强制使用严格的 KeyPrefix：
- `tts:speaker:{id}`: 缓存 Speaker Embedding (张量)。
- `tts:session:{id}`: 缓存对话过程中的流式状态。
- **约束**：严禁在 `MOLT_BRAIN` (DB 0) 或 `RAG` 的 Redis 空间内写入。

---

## 6. 接口设计（Agent 调用）

### 6.1 生成接口
`POST /tts`
```json
{
  "text": "...",
  "emotion": {"valence": 0.5, "arousal": 0.8},
  "speaker_id": "global_default",
  "stream": true
}
```

### 6.2 克隆接口
`POST /tts/clone`
```json
{
  "audio_base64": "...",
  "user_id": "...",
  "alias": "my_clone_voice"
}
```

---

## 7. 质量验收 (Acceptance Criteria)

### 7.1 自动评估拦截 (Quality Gate)
- **WER (字错率)**: < 10%
- **DNSMOS (音质)**: Overall ≥ 4.0
- **Speaker Similarity**: Cosine ≥ 0.82
- **失败决策**：高于/低于阈值时，自动拦截输出并向 Agent 返回异常信号。

---

## 8. 禁忌 (Do NOT)
1. **禁止混库**：严禁在 Brain (MOLT) 或 RAG 的 Collection 中插入 TTS 条目。
2. **禁止硬编码设备**：必须使用 `utils/device.py` 自动检测，禁止出现 `device="cuda"` 等硬编码。
3. **禁止同步等待克隆**：克隆逻辑应支持异步处理或高效缓存。
4. **禁止忽略呼吸感**：后处理必须包含随机的微小 jitter 和呼吸音插入。

---

## 9. 资源获取 (Resource Acquisition)

为确保环境一致性，请按以下路径同步源码与模型权重：

### 9.1 源码同步 (Git Submodule)
建议使用 submodule 方式引入，保持第三方库独立更新：
- **仓库地址**：`https://github.com/yl4579/StyleTTS2.git`
- **执行命令**：
  ```bash
  git submodule add https://github.com/yl4579/StyleTTS2.git third_party/StyleTTS2
  pip install -r third_party/StyleTTS2/requirements.txt
  ```

### 9.2 预训练模型下载 (Model Weights)
请将下载的权重文件统一放置于项目 `models/tts/` 目录下（配置见 `constants.py`）：

| 模型类型 | 下载链接 (HuggingFace) | 推荐用途 |
| :--- | :--- | :--- |
| **LJSpeech (24kHz)** | [yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) | 单人高保真、基础情感 |
| **LibriTTS (Multi-speaker)** | [yl4579/StyleTTS2-LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS) | **推荐** 多人克隆、VAD情感增强 |
| **Speaker Encoder** | [LibriTTS weights 内部集成] | 提取语音特征用于克隆 |

### 9.3 依赖安装提示
同步源码后需安装特定依赖（注意版本冲突）：
- `torchaudio` (必须与 PyTorch 版本配套)
- `phonemizer` (处理文本音素化)
- `Munch` (StyleTTS2 内部配置管理)
- `vector-quantize-pytorch`