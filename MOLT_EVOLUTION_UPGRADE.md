# MOLT Brain 进化升级计划 (MOLT_EVOLUTION_UPGRADE)

本项目将借鉴 OpenClaw Foundry 的“工程化演进”思路，通过结合已有的 `skill-creator` 技能和原生 MOLT Brain 的反馈机制，实现 Agent 技能的自动研究、编写与热部署。

## 1. 进化循环工作流 (Evolution Workflow)

我们将实现一个闭环的自演进流程：

1.  **Observe (观察)**：`AgentRuntime` 在执行过程中记录详细的 Trace（目标、工具调用、原始结果、上下文）。
2.  **Feedback (反馈信号)**：`FeedbackJudge` 识别出负反馈（任务失败）或规划器识别到知识缺失。
3.  **Research (研究)**：`EvolutionDispatcher` 指派研究任务。自动抓取相关技术文档或 GitHub 库，并将其转化为结构化的 `LTM (长期记忆)` 节点。
4.  **Learn (模式总结)**：对历史 Trace 和研究结论进行抽象，生成 **Skill IR (领域中间表示)**，描述新技能的功能、输入输出及核心逻辑。
5.  **Write (编写)**：调用系统中已有的 **`skill-creator`** 技能。
    - **输入**：Skill IR + 研究素材。
    - **逻辑**：由 `skill-creator` 负责生成符合规范的 Python 代码、`SKILL.md` 元数据及单元测试。
6.  **Deploy (部署)**：
    - `SkillRegistry` 执行热加载逻辑，加载 `skills/generated/` 下的新模块。
    - `EvolutionDispatcher` 通知 `PlanningAgent` 新能力已上线。
7.  **Verify (验证)**：在下一轮同类任务中尝试新技能，记录结果并反馈至 MOLT Brain 以调整权重。

---

## 2. 关键组件增强

### A. Trace 增强 (Observe)
优化 `agents/agent_runtime.py` 的历史记录，记录：
- `tool_call_payload`: 完整的工具参数。
- `raw_observation`: 原始执行结果（未被 LLM 总结前的数据）。
- `thought_process`: Agent 在调用前的推理逻辑。

### B. 自动化研究员 (SkillResearcher)
在 `services/` 目录下新增 `skill_researcher.py`：
- **功能**：基于互联网或 RAG 检索，提取 API 接口定义、代码示例和最佳实践。
- **产出**：注入到 MongoDB 的 `M3_Episodic` 层，并关联到特定的任务标签。

### C. 技能生成适配器 (Skill-Creator Bridge)
与其让 LLM 直接写文件，不如规范化对 `skill-creator` 的调用：
- 定义一套 `Skill-IR` 标准，包含：`name`, `purpose`, `dependencies`, `api_schema`, `logic_pseudo_code`。
- 将 IR 作为提示词输入给 `skill-creator`。

### D. 注册器热更新 (Hot Reloading)
为 `skills/skill_registry.py` 增加动态加载方法：
- `refresh_generated_skills()`: 专门重新扫描 `skills/generated/` 目录。
- 确保模块更新不会导致主进程崩溃（使用隔离加载测试）。

---

## 3. 实施路径

### 阶段 1：Trace 结构化与触发器联调
- 修改 `AgentRuntime` 以支持更高精细度的日志记录。
- 在 `EvolutionDispatcher` 中定义触发“技能研发”的置信度阈值。

### 阶段 2：研究与 IR 生成逻辑
- 开发 `SkillResearcher` 服务，对接搜索工具。
- 实现从“失败记录 + 文档”到 “Skill IR” 的总结 Prompt。

### 阶段 3：`skill-creator` 自动化集成
- 编写脚本，允许 `EvolutionDispatcher` 直接以编程方式触发 `skill-creator` 任务。
- 配置 `skills/generated/` 路径的自动索引。

### 阶段 4：验证与闭环
- 实现新技能的“试用期”评价机制（Alpha Test）。
- 建立成功率对比基准，验证进化后的性能提升。

---

## 4. 优势
- **复用性**：最大化利用项目中现有的 `skill-creator` 和 `SkillRegistry`。
- **可解释性**：所有进化的技能首先以 IR 形式被人类或判别模型审核，代码生成在后。
- **原生兼容**：新技能与普通技能在同一套 `MOLT_BRAIN` 权重体系下运行。
