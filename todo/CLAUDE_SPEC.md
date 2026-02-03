# Agent 加载 Skill + MCP 方案（参考 Claude，全局可由环境变量指定）
 
## 背景与目标
- 参考 Claude 的设计，统一 Agent 对技能（Skills）与 MCP 工具的加载方式
- 支持通过环境变量指定“Claude 全局”技能与 MCP 连接的加载来源
- 在不配置环境变量时保持当前行为（仅加载项目内 skills/ 下内容），确保向后兼容
 
## 设计原则
- 最小侵入：优先在现有注册流程上扩展，避免大范围重构
- 明确边界：技能（Skill）与 MCP 工具（Tool）均作为“可发现能力”注入 Agent
- 显式配置：通过环境变量明确启用与路径来源，避免隐式耦合到平台目录结构
- 安全可控：工具注册时带权限名，配合 PermissionManager 做审批
- 能力统一：Skills 与 MCP 以统一的 Capability 抽象暴露给 Planner
 
## 环境变量约定
- LLP_ENABLE_CLAUDE_GLOBAL=1 开启从“Claude 全局”位置加载
- LLP_SKILLS_DIRS=/path1:/path2 额外技能目录，冒号分隔
- LLP_CLAUDE_SKILLS_DIR=/abs/path 指向 Claude 全局技能根目录（包含若干技能子目录或 skills 子目录）
- LLP_MCP_CONNECTIONS_DIR=/abs/path 指向 MCP 连接定义目录（每个连接一个 JSON/YAML 文件）
- LLP_MCP_CONNECTIONS_FILE=/abs/path 指向 MCP 连接汇总文件（JSON/YAML，集中定义多连接）
- LLP_MCP_DEFAULT_SAFETY=MEDIUM 默认权限安全级别，支持 LOW|MEDIUM|HIGH|CRITICAL
 
## 加载流程
- Skills 加载
  - 启动时读取 LLP_SKILLS_DIRS 与 LLP_CLAUDE_SKILLS_DIR
  - 遍历每个目录，优先尝试 Claude 风格 SKILL.md 前言解析；否则尝试 Python 模块或包内 SKILL_MANIFEST/register_skill
  - 将解析得到的 SkillManifest 注册到全局 SkillRegistry
- MCP 加载
  - 通过 LLP_MCP_CONNECTIONS_DIR 或 LLP_MCP_CONNECTIONS_FILE 读取连接配置，逐个连接启动 MCP 服务进程
  - 当前实现：按配置中的 tools 静态注册到 MCPTaskAgent（mcp.<tool_name> 命名）
  - 计划增强：接入 MCP initialize/tools.list 协议做动态发现与健康检查
  - 工具注册时绑定权限名与默认/覆盖的安全级别，接入 PermissionManager
 
## 代码改动点
- skills/skill_registry.py
  - 扩展 init_skills：增加对环境变量的读取与目录合并，支持冒号分隔多路径
  - 对每个路径若存在子目录 skills/，优先从该子目录加载（兼容 Claude 目录结构）
  - 维持 parse_claude_skill 针对 SKILL.md 的前言解析流程
  - 相关位置参考 [skill_registry.py](file:///Volumes/Extra/Projects/local-llm-provider/skills/skill_registry.py)
- agents/agent_runtime.py
  - 在创建 MCPTaskAgent 后，调用 MCP 加载器从环境变量注册工具
  - 将可用技能与 MCP 工具写入 runtime.state.context
  - 新增：统一 Capability 列表与 planning_hints（包含安全等级与审批需求）
  - 相关位置参考 [agent_runtime.py](file:///Volumes/Extra/Projects/local-llm-provider/agents/agent_runtime.py)
- agents/task_agents/mcp_agent.py
  - 保持 register_tool/get_available_tools/权限检查逻辑不变
  - 增加与 MCP 加载器的协作接口（由加载器调用 register_tool 完成注册）
  - 相关位置参考 [mcp_agent.py](file:///Volumes/Extra/Projects/local-llm-provider/agents/task_agents/mcp_agent.py)
- 新增：utils/mcp_loader.py（最小必要新增）
  - load_from_env(mcp_agent)：
    - 解析 LLP_MCP_CONNECTIONS_DIR/FILE（支持 JSON/YAML）
    - 启动每个 MCP 服务（command/args/env/workdir），心跳监测进程状态（READY/DEAD）
    - 将配置中的 tools 按 mcp.<tool_name> 注册到 mcp_agent（可配置安全级别）
    - 计划增强：协议 initialize/tools.list 动态发现与退化标记（DEGRADED）
  - 提供停止/回收进程的钩子，防止孤儿进程（已在 main.py shutdown 钩子接入）
  - 相关位置参考 [mcp_loader.py](file:///Volumes/Extra/Projects/local-llm-provider/utils/mcp_loader.py)
- 新增：schemas/capability.py
  - 定义统一的 Capability 抽象（SKILL/MCP），含安全等级与审批提示
  - 相关位置参考 [capability.py](file:///Volumes/Extra/Projects/local-llm-provider/schemas/capability.py)
 
## 配置格式（示例）
```yaml
# 单文件示例（LLP_MCP_CONNECTIONS_FILE）
connections:
  - name: chrome_devtools
    command: /usr/local/bin/chrome-mcp
    args: ["--headless"]
    env:
      CHANNEL: stable
    workdir: /tmp
    safety: HIGH
    tools: ["navigate_page", "take_screenshot"]
  - name: web_search
    command: /usr/local/bin/web-search-mcp
    args: []
    safety: MEDIUM
    tools: ["web_search"]
```
 
## 权限与安全
- 权限名：统一采用 mcp.<tool_name>，便于 PermissionManager 做细粒度审批
- 安全级别：来源于连接项 safety 字段，默认 LLP_MCP_DEFAULT_SAFETY
- 双阶段权限：
  - Planning 阶段：超过阈值的能力标记 requires_approval，Planner 可见提示
  - Execution 阶段：PermissionManager 确认审批，避免执行中断
  - 相关位置参考 [agent_runtime.py](file:///Volumes/Extra/Projects/local-llm-provider/agents/agent_runtime.py)、[permission_manager.py](file:///Volumes/Extra/Projects/local-llm-provider/permission_manager.py)
 
## 兼容与回退
- 未设置环境变量时：
  - Skills：仅加载项目内 skills/ 与 skills/claude-skills/skills（现有行为）
  - MCP：仅有默认 read_file 工具，保持现状
- 配置错误/目录不存在时：
  - 跳过该路径并记录日志，不影响其他路径与现有功能
 
## 测试计划
- 单元测试
  - Skills：模拟 LLP_SKILLS_DIRS 与 LLP_CLAUDE_SKILLS_DIR，验证 SkillRegistry 列表包含外部技能
  - MCP：模拟连接文件，注入假 MCP 服务（子进程立即退出），验证不会注册工具
    - 相关位置参考 [test_mcp_failure_injection.py](file:///Volumes/Extra/Projects/local-llm-provider/tests/test_mcp_failure_injection.py)
  - 参考现有测试用例结构，可新增 tests/test_mcp_env_loading.py、tests/test_skills_env_loading.py
- 集成测试
  - 启动 main.py，设置环境变量后命中 PlanningAgent 的能力发现路径，检查 available_skills/available_mcp_tools
  - 验证 planning_hints 中的 requires_approval 提示被 Planner 正确使用
 
## 迁移步骤
- [x] 第一步：实现 utils/mcp_loader.py 与 skill_registry.py 的环境变量扩展
- [x] 第二步：在 agent_runtime.py 注入加载流程，并填充 runtime.state.context
- [x] 第三步：新增测试用例，完善 .env.example（仅文档中说明，不影响当前提交）
- [x] 第三步补充：验证全局技能自动加载（tests/verify_global_skills_loading.py）
- [x] 第四步：验证在 macOS/Linux/WSL 环境下的路径与进程管理稳定性
- [ ] 第五步（增强）：实现 MCP 协议 initialize/tools.list 与退化策略（DEGRADED）
 
## 示例环境变量
```bash
export LLP_ENABLE_CLAUDE_GLOBAL=1
export LLP_SKILLS_DIRS="/Users/me/custom-skills:/opt/shared/skills"
export LLP_CLAUDE_SKILLS_DIR="/Users/me/Library/Application Support/Claude/skills"
export LLP_MCP_CONNECTIONS_FILE="/Users/me/.claude/mcp_connections.yaml"
export LLP_MCP_DEFAULT_SAFETY="MEDIUM"
```
 
## 备注
- 方案确保“可发现能力”统一暴露给 PlanningAgent，支持优先使用已存在的技能与工具
- 若某能力不存在，可通过技能机制补齐（例如 skill-creator），不与 MCP 直接耦合
 - 通过统一 Capability 与 planning_hints，支持后续成本评估与自动 fallback
 
