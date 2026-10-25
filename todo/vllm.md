# vLLM 集成方案设计文档

## 背景

当前项目使用三种推理引擎：
- **TransformersEngine** - Hugging Face Transformers 原生推理，支持所有架构（含 `trust_remote_code` 自定义架构）
- **LlamaCppEngine** - llama.cpp 服务器模式，专用于 GGUF 量化模型
- **ONNX** - 未实现

决策逻辑仅基于文件后缀（`UnifiedModelLoader.__init__` 第28-33行），不检查模型架构兼容性。

vLLM 作为高性能推理引擎，具有 PagedAttention、连续批处理、Tensor Parallel 等核心优势，但受限于：
1. 仅支持已在代码库注册的架构（`config.json` 的 `architectures` 字段必须匹配）
2. GGUF 支持实验性且优化不足
3. 强依赖 CUDA，Mac/CPU/AMD 支持较弱
4. 自定义架构（需 `trust_remote_code`）无法直接使用

因此采用**三引擎共存 + 智能路由**策略，而非替换现有引擎。

---

## Phase 1：最小侵入式集成（当前实施）

### 目标
- 新增 `VllmEngine` 实现 `InferenceEngine` 接口
- 在 `UnifiedModelLoader` 中增加智能路由逻辑
- 以可选依赖方式引入，零破坏现有功能
- 通过环境变量/API 参数控制引擎选择

### 实施内容

#### 1. 新增文件：`model_providers/vllm_engine.py`
- 实现 `InferenceEngine` 抽象基类
- 封装 `AsyncLLMEngine` 提供流式生成
- 支持张量并行、量化（AWQ/GPTQ/FP8）、前缀缓存等配置

#### 2. 修改 `model_providers/unified_model_loader.py`
- 新增 `engine_preference` 配置：`auto` | `vllm` | `transformers` | `llama_cpp`
- 新增 `_load_hf_model_smart()` 智能路由逻辑
- 新增 `_try_load_vllm()` 尝试加载并回退
- 新增 `_is_vllm_compatible()` 架构兼容性预检查（读取 `config.json` 的 `architectures` 字段）

#### 3. 依赖管理
- `pyproject.toml` 增加 `extras_require` 可选依赖：`vllm>=0.6.0`
- 不修改主依赖列表，避免强制安装

#### 4. 配置项
```bash
# 环境变量
ENGINE_PREFERENCE=auto              # auto | vllm | transformers | llama_cpp
VLLM_TENSOR_PARALLEL=1              # 张量并行度
VLLM_GPU_MEMORY_UTIL=0.9            # GPU 显存利用率
VLLM_QUANTIZATION=awq               # awq | gptq | fp8 | None
VLLM_MAX_MODEL_LEN=8192             # 最大序列长度
VLLM_ENFORCE_EAGER=false            # 禁用 CUDA Graph（调试用）

# API 传参
options = {
    "engine": "vllm",
    "vllm_options": {
        "tensor_parallel_size": 2,
        "quantization": "awq",
        "max_model_len": 32768,
    }
}
```

---

## Phase 2：生产化增强（后续规划）

### 目标
将 vLLM 从库模式升级为生产级服务部署模式。

### 实施内容

#### 1. vLLM 服务器化部署（OpenAI 兼容 API 模式）
- 独立进程运行 `vllm serve`，通过 HTTP API 调用
- 解决库模式下的内存碎片、进程隔离、热重载问题
- 支持多实例负载均衡

#### 2. 模型预热与健康检查
- 启动时自动预热（dummy input 触发 CUDA Graph 编译）
- `/health` 端点监控引擎状态
- 就绪探针/存活探针支持 K8s 部署

#### 3. 指标监控
- 集成 Prometheus metrics（`/metrics` 端点）
- 关键指标：吞吐率、延迟分位数、GPU 利用率、队列长度、缓存命中率
- Grafana 仪表板模板

#### 4. 自动 Tensor Parallel 规划
- 根据 GPU 数量、显存大小、模型参数量自动计算最优 TP 度
- 支持 Pipeline Parallel（超大模型跨节点）

#### 5. Prefix Caching 深度配置
- 系统提示词自动缓存复用
- 缓存键自定义（支持多租户隔离）
- 缓存驱逐策略配置

#### 6. 请求级调度优化
- 优先级队列（系统/付费/免费用户分级）
- 最大等待时间控制
- 批处理大小动态调整

### 原因
- 库模式适合开发测试，生产环境需进程隔离、故障隔离、滚动更新
- 监控指标是 SLA 保障基础
- 自动化运维降低人力成本

### 优点
- 生产级稳定性、可观测性、可运维性
- 支持大规模并发、多模型热加载
- 标准化 OpenAI API，客户端零改动迁移

### 缺点
- 部署复杂度显著上升（需容器编排、服务发现、负载均衡）
- 网络开销增加（HTTP vs 进程内调用）
- 资源占用基线提高（独立进程开销）

---

## Phase 3：长期演进（视情况考虑）

### 触发条件（需全部满足）
- [ ] vLLM 原生支持覆盖 99%+ 目标模型架构
- [ ] ROCm/MPS/CPU 后端达到 CUDA 同等成熟度
- [ ] GGUF 支持性能达到 llama.cpp 90%+ 水平
- [ ] 团队不再需要 `trust_remote_code` 灵活加载自定义架构

### 可能动作
- 逐步淘汰 TransformersEngine（保留作为调试工具）
- 统一迁移至 vLLM 服务器化架构
- 移除 llama.cpp 依赖（仅保留 GGUF 转换工具链）

---

## 三引擎定位对比表

| 维度 | LlamaCppEngine | TransformersEngine | VllmEngine (新增) |
|------|---------------|-------------------|------------------|
| **核心定位** | GGUF 专用 / 边缘部署 | 通用兜底 / 调试 / 自定义架构 | 高性能生产 / 高并发 / 标准架构 |
| **支持格式** | .gguf 单文件 | HF 目录 (config.json + .safetensors) | HF 目录 + 量化格式 (AWQ/GPTQ/FP8) |
| **架构兼容性** | 依赖 llama.cpp 实现 | **100%**（trust_remote_code） | 仅 vLLM 注册架构 (~50+ 主流) |
| **硬件支持** | CPU / Apple Metal / CUDA / ROCm | CPU / CUDA / MPS / ROCm | **CUDA 为主**，ROCm 实验性 |
| **并发能力** | 低（单请求串行） | 低（无连续批处理） | **极高**（连续批处理 + PagedAttention） |
| **吞吐率基准** | 1x | 1x | **5-30x** |
| **显存效率** | 中等 | 低（静态分配） | **高**（PagedAttention 动态） |
| **多卡扩展** | 不支持 | device_map 简易分片 | **Tensor Parallel 原生线性扩展** |
| **量化支持** | GGUF 全谱系 | bitsandbytes 4/8bit | AWQ / GPTQ / FP8 / Marlin (推理加速) |
| **冷启动** | 快 | 中等 | 慢（需内核编译） |
| **部署模式** | 库/服务器均可 | 库模式 | 库模式 / 独立服务器 |
| **适用场景** | 个人开发、边缘设备、量化模型分发 | 调试、自定义架构、CPU推理、Mac | 生产API服务、高并发、长上下文、多卡 |

---

## 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-10-25 | 采用三引擎共存 + 智能路由 | 单一引擎无法覆盖所有场景，vLLM 约束过多无法完全替代 |
| 2026-10-25 | Phase 1 以可选依赖引入 | 零破坏现有用户，按需安装 |
| 2026-10-25 | 架构兼容性检查读取 config.json | 避免运行时报错，快速失败回退 |
| 2026-10-25 | 保留 LlamaCppEngine 处理 GGUF | vLLM GGUF 支持实验性，llama.cpp 是 GGUF 金标准 |

---

## 相关文件清单

### 新增
- `model_providers/vllm_engine.py` - vLLM 引擎实现

### 修改
- `model_providers/unified_model_loader.py` - 智能路由逻辑
- `pyproject.toml` - 可选依赖配置

### 配置
- `.env.example` - 环境变量示例（需新增）

---

## 测试验收标准

### Phase 1 验收
- [ ] `pip install -e .[vllm]` 成功安装 vLLM 依赖
- [ ] 标准架构模型（如 Qwen2.5-7B、Llama-3.1-8B）可通过 `engine=vllm` 加载并推理
- [ ] `engine=auto` 在 vLLM 不可用时自动回退 TransformersEngine
- [ ] 自定义架构模型（需 trust_remote_code）在 `engine=auto` 时正确回退
- [ ] GGUF 模型仍使用 LlamaCppEngine，不受影响
- [ ] 多卡环境下 `tensor_parallel_size>1` 正常工作
- [ ] 量化模型（AWQ/GPTQ）正确加载并推理

### Phase 2 验收（规划中）
- [ ] vLLM 独立服务器模式启动成功
- [ ] OpenAI 兼容 API 正常响应
- [ ] Prometheus metrics 暴露正常
- [ ] 健康检查端点返回就绪状态
- [ ] 预热流程自动执行
- [ ] 滚动更新零停机