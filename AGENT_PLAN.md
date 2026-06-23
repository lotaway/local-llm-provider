# Agent Client Protocol 整合计划 (已实施)

## 1. 最终架构

```
┌──────────────────────────────────────────────────┐
│          后端 HTTP 服务 (local-llm-provider)       │
│                                                   │
│  原始能力层:                                       │
│  ┌──────────┐ ┌──────┐ ┌────────┐ ┌──────────┐   │
│  │ LLM 推理  │ │ RAG  │ │ 记忆系统 │ │ Skills   │   │
│  └──────────┘ └──────┘ └────────┘ └──────────┘   │
│                                                   │
│  ACP 适配器 (供外部 ACP 客户端, 如 Zed):             │
│  ┌──────────────────────────────────────────────┐  │
│  │  ProtocolBridge (ACP Agent)                  │  │
│  │  ↓ 直接调用 LocalLLModel.chat()               │  │
│  │  ↓ 不涉及 AgentRuntime / 个体 Agent            │  │
│  └──────────────────────────────────────────────┘  │
│                                                   │
│  向后兼容:                                         │
│  ┌──────────────────────────────────────────────┐  │
│  │  controllers/agent_controller.py             │  │
│  │  (DEPRECATED, 保留但标记废弃)                  │  │
│  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
          │                           ▲
          │ HTTP API                  │ ACP stdio
          ▼                           │
┌──────────────────────┐    ┌─────────┴──────────────┐
│ 外部 HTTP 调用方      │    │ CLI 客户端 (llp-cli)    │
│                      │    │                        │
│ 自行实现：            │    │ AgentRuntime 执行循环    │
│ - AgentRuntime 循环  │    │ QA / Planning / Router  │
│ - 用户交互            │    │ Verification / Risk     │
│ - 文件操作            │    │ Error Handler 等 Agent  │
│                      │    │ RemoteLLM (HTTP→后端)   │
│                      │    │                        │
│ 调用后端：            │    │ /add <path>            │
│ - LLM 推理 API       │    │ /add-dir <path>        │
│ - RAG API            │    │ /context / /clear      │
│ - 记忆 API           │    │ 文本输入 → 执行 Agent   │
└──────────────────────┘    └────────────────────────┘
```

## 2. 核心设计原则

| 原则 | 实现方式 |
|------|---------|
| **执行层客户端化** | AgentRuntime + 所有个体 Agent 移入 `clients/agents/` |
| **后端纯服务化** | 后端只暴露原始能力（LLM / RAG / 记忆 / Skills），不包含执行循环 |
| **CLI 类 opencode** | CLI 客户端管理用户交互、文件上下文、AgentRuntime 执行循环 |
| **ACP 兼容** | 后端 `ProtocolBridge` 实现 ACP Agent 接口，供外部工具（Zed）集成 |
| **HTTP 向后兼容** | 保留 `/agents/*` 端点但标记 deprecated |

## 3. 组件职责

### 3.1 后端服务（server side）

| 组件 | 位置 | 职责 |
|------|------|------|
| **ModelProvider** | `model_providers/` | LLM 推理引擎（本地/远程） |
| **RAG** | `rag.py` | 文档检索增强生成 |
| **Memory System** | `services/` | 三层记忆（M1感觉/M2工作/M3稳定） |
| **Skills** | `skills/` | Claude Skill 兼容、MCP 工具加载 |
| **ACP ProtocolBridge** | `adapters/protocol_bridge.py` | ACP Agent 接口实现，直接调用 LLM |
| **HTTP Controllers** | `controllers/` | REST API 端点（LLM / RAG / 记忆） |

### 3.2 CLI 客户端（client side）

| 组件 | 位置 | 职责 |
|------|------|------|
| **AgentRuntime** | `clients/agents/agent_runtime.py` | 多 Agent 调度、状态管理、执行循环 |
| **BaseAgent** | `clients/agents/agent_base.py` | Agent 基类 + AgentResult/AgentStatus |
| **QA Agent** | `clients/agents/qa_agent.py` | 问题解析与意图识别 |
| **Planning Agent** | `clients/agents/planning_agent.py` | 任务规划与分解 |
| **Router Agent** | `clients/agents/router_agent.py` | Agent 路由分发 |
| **Verification Agent** | `clients/agents/verification_agent.py` | 结果验证与质量控制 |
| **Risk Agent** | `clients/agents/risk_agent.py` | 风险评估与安全检查 |
| **Error Handler** | `clients/agents/error_handler_agent.py` | 异常处理与自动恢复 |
| **Task Agents** | `clients/agents/task_agents/` | LLM / RAG / MCP 任务执行 |
| **RuntimeFactory** | `clients/agents/runtime_factory.py` | AgentRuntime 工厂方法 |
| **RuntimeState** | `clients/agents/runtime_state.py` | 运行时状态管理 |
| **ContextStorage** | `clients/agents/context_storage.py` | 会话上下文持久化 |
| **RemoteLLM** | `clients/remote_llm.py` | LLM HTTP 代理（用 httpx 调用后端） |
| **CLI Entry** | `clients/cli.py` | 交互式命令行入口 |

### 3.3 向后兼容层

| 组件 | 位置 | 说明 |
|------|------|------|
| **agents/ stub** | `agents/__init__.py` | 从 `clients.agents` 再导出，保持旧导入兼容 |
| **agent_controller.py** | `controllers/agent_controller.py` | 端点保留但启动时 log DEPRECATED 警告 |

## 4. 通信流程

### CLI 客户端 → 后端通信
```
User Input → clients/cli.py
  → AgentRuntime.execute()
    → QAAgent.execute() → RemoteLLM.chat(messages)
      → HTTP POST /v1/chat/completions → LocalLLModel.chat()
      ← SSE stream ← response chunks
    → PlanningAgent.execute() → RemoteLLM.chat(...) ...
  → 最终结果输出到终端
```

### 外部 ACP 客户端通信
```
Zed (ACP Client) → ACP stdio
  → ProtocolBridge (ACP Agent)
    → LocalLLModel.chat()  ← 直接调用，无 AgentRuntime
  ← PromptResponse ←
```

### 外部 HTTP 调用方通信
```
外部应用 → HTTP POST /v1/chat/completions
  → LocalLLModel.chat()
← SSE stream / JSON response ←

外部应用需要自行实现 AgentRuntime 执行循环和 Agent 逻辑。
```

## 5. 目录结构

```
project/
├── adapters/                    # ACP 适配器（后端）
│   ├── base_adapter.py          # TransportAdapter 抽象接口
│   ├── protocol_bridge.py       # ACP Agent 实现 → 直接调用 LLM
│   ├── stdio_adapter.py         # stdio 入口（acp-agent 命令）
│   └── http_adapter.py          # HTTP 适配器（存根）
├── agents/                      # 向后兼容层（stub）
│   └── __init__.py              # 从 clients.agents 再导出
├── clients/                     # 客户端（执行层）
│   ├── agents/                  # AgentRuntime + 所有 Agent
│   │   ├── agent_runtime.py
│   │   ├── agent_base.py
│   │   ├── qa_agent.py
│   │   ├── planning_agent.py
│   │   ├── router_agent.py
│   │   ├── verification_agent.py
│   │   ├── risk_agent.py
│   │   ├── error_handler_agent.py
│   │   ├── runtime_factory.py
│   │   ├── runtime_state.py
│   │   ├── context_storage.py
│   │   ├── error_utils.py
│   │   └── task_agents/
│   │       ├── llm_agent.py
│   │       ├── rag_agent.py
│   │       └── mcp_agent.py
│   ├── remote_llm.py            # LLM HTTP 代理
│   └── cli.py                   # 交互式 CLI 入口
├── controllers/                 # HTTP 控制器
│   └── agent_controller.py      # (DEPRECATED)
├── schemas/
│   └── execution_protocol.py    # 内部协议定义
├── model_providers/             # LLM 推理引擎
├── services/                    # 记忆/反馈/演化
├── skills/                      # Claude Skill 管理
├── main.py                      # HTTP 服务入口
└── pyproject.toml               # 入口点配置
```

## 6. 入口点

| 命令 | 作用 | 启动方式 |
|------|------|---------|
| `local-llm-provider` | 后端 HTTP 服务（独立运行） | `python main.py` |
| `llp-cli` | CLI 交互式客户端 | `python -m clients.cli` |
| `acp-agent` | ACP stdio 模式 | `python -m adapters.stdio_adapter` |

CLI 通过 `LLP_BACKEND_URL` 环境变量或默认 `http://localhost:8434` 连接后端。

## 7. 已实施变更

### 文件迁移
- `agents/*.py` → `clients/agents/`（保留 `agents/__init__.py` 作为向后兼容 stub）

### 新增文件
| 文件 | 行数 | 说明 |
|------|------|------|
| `schemas/execution_protocol.py` | 32 | ExecutionRequest/Response/Status 定义 |
| `adapters/base_adapter.py` | 24 | TransportAdapter 抽象接口 |
| `adapters/protocol_bridge.py` | 140 | ACP Agent，直接调用 LocalLLModel（无 AgentRuntime） |
| `adapters/stdio_adapter.py` | 16 | ACP stdio 启动入口 |
| `adapters/http_adapter.py` | 18 | HTTP 适配器存根（NotImplementedError） |
| `clients/remote_llm.py` | 60 | 通过 HTTP 调用后端 LLM 的代理 |
| `clients/cli.py` | 138 | 交互式 CLI，支持 `/add` `/add-dir` `/context` `/clear` |

### 修改文件
| 文件 | 变更 |
|------|------|
| `controllers/agent_controller.py` | 导入路径改为 `clients.agents.*`，添加 DEPRECATED 日志 |
| `services/task_scheduler.py` | 导入路径改为 `clients.agents.*` |
| `services/skill_researcher.py` | 导入路径改为 `clients.agents.*` |
| `tests/*.py` | 导入路径改为 `clients.agents.*` |
| `main.py` | 保留 `--stdio` 标志 |
| `pyproject.toml` | 添加 `clients` 包和 `llp-cli` 入口点 |

## 8. 关键设计决策

### 为什么 AgentRuntime 移到客户端？

AgentRuntime 是执行循环，涉及：
- 用户交互（输入、确认）
- 文件系统操作（读/写文件）
- 多 Agent 编排决策

这些天然适合在客户端执行。后端保持为无状态服务层，只提供原始能力。

### RemoteLLM 的设计

`RemoteLLM` 实现了与 `LocalLLModel` 相同的接口（`chat()`、`chat_at_once()`、`format_messages()`、`extract_thought()`等），使 `clients/agents/` 中的 Agent 代码无需修改即可透明地通过 HTTP 调用后端 LLM。

### ProtocolBridge 简化

后端的 `ProtocolBridge` 不再包装 AgentRuntime，而是直接调用 `LocalLLModel.chat()`。其目的是让外部 ACP 客户端（如 Zed）可以直接与后端交互，但不提供多 Agent 编排能力。

## 9. 向后兼容策略

| 场景 | 策略 |
|------|------|
| `from agents import AgentRuntime` | 通过 `agents/__init__.py` 从 `clients.agents` 再导出 |
| `from agents.qa_agent import QAAgent` | 同样通过再导出（已改为 `from clients.agents.qa_agent`） |
| `GET /v1/agents/metadata` | 端点保留 |
| `POST /v1/agents/run` | 端点保留，启动时打印 DEPRECATED 警告 |
| 现有测试 | 导入路径更新为 `clients.agents.*` |

## 10. 测试状态

| 测试模块 | 状态 | 说明 |
|---------|------|------|
| `tests/test_execution_protocol.py` | 5 passed | 内部协议模型 |
| `tests/test_agent_protocol.py` | 14 passed | 原协议类型 |
| `tests/test_protocol_bridge.py` | 11 passed | 新 ACP Agent 桥接 |
| `tests/test_http_adapter.py` | 3 passed | HTTP 适配器存根 |
| `tests/test_agent_system.py` | 1 skipped | 需模型环境 |
| `tests/test_private_context.py` | 1 skipped | 需 pytest-asyncio |
| **总计** | **33 passed** | |

## 11. 入口点配置

```toml
[project.scripts]
local-llm-provider = "main:main"
llp-cli = "clients.cli:main"
acp-agent = "adapters.stdio_adapter:main"
```
