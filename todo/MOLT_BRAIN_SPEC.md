0. 总体目标（升级版）

Agent 的记忆系统必须满足：

分层记忆 + 生命周期机制

遗忘与强化都是可控 / 可解释

自主评价 & 自我调整策略（Molt 层）

支持多 Agent 共享 / 私有 /共识知识

存储有意义的结构化知识，而非堆积 Context

这个架构融合了研究上的“多级记忆分层模型”理念（类似层次化缓存/分层记忆）及 Agent 记忆系统设计最佳实践。

🧱 1. 记忆层级 与 Molt 触发规则

每层不仅定义存储与淘汰规则，还定义：

何时被 Molt 系统评估 & 可能调整行为规则（Prompt / 索引策略 /聚合方式）

🔹 M1：感觉缓存（Sensory / Immediate）

原则：只感知，不记忆
目标：快速接收当前输入，极短期临时存储
执行：

存储：MongoDB + TTL Index

字段：timestamp / source / attention_score / raw_content

默认 TTL：很短

Molt 触发检查（单次即时）：

当 raw_content 被重复访问（触发 attention_score 上升）时 → 标记“有用输入信号”

如果注意力、任务相关性高 → 触发进入 M2 判断

禁止：

不向量化

不参与长期索引

不允许被 Agent 主动回忆

🔹 M2：工作记忆（Working Memory）

原则：推理/规划/短期上下文
目标：支撑当前任务，严格容量限制

结构：

{
  "content",
  "created_at",
  "last_accessed",
  "rehearsal_count",
  "links_to_other_chunks"
}


Molt 触发规则（实时）：

当同一 “content chunk” 被多次引用（rehearsal_count 增加）
→ 发出“潜在长期价值”信号

当低价值或过时内容导致推理误导
→ Molt 层触发 Prompt/Chunk 生成规则微调

淘汰：

超过容量上限 → 基于 rehearsal_count & recency 淘汰

Agent 进程重启清空

🔹 M3：短期记忆（Short-term / Episodic）

目标：被 Agent 回忆 & 可能强化
存储：

Milvus/ES 向量

Neo4j 情境图

MongoDB 原始片段

Molt 触发判断：

定期评估记忆权重 / 召回命中情况
→ 自动调整记忆权重 decay schedule

当某一记忆在多个情境下被检索并强化
→ 自主参与 “压缩 & 抽象” 进入 LTM

实现细节：

embedding + semantic_summary + importance + graph_links

复习行为会延缓遗忘

权重可解释、可追踪

🔹 M4：长期记忆（Long-term Memory，LTM）

目标：高质量抽象（避免无脑对话堆积）

存储：

存储	内容
Neo4j	概念关系、因果图
Milvus	语义召回向量
ES	tag / keyword 辅助检索

压缩规则（自动）：

多条相关 M3 → 抽象成一个 LTM 节点

只保留：结论 / 适用条件 / 置信度

Molt 触发（定期 & 事件）：

当大量 M3 指向同一主题时
→ 自动触发“合并与抽象器”

当冲突知识出现
→ 标记并保持版本化（不覆盖）

🔹 M5：程序性记忆（Procedural Memory）

目标：记录可复用策略 / 流程（不是对话）

结构：

step_pattern

preconditions / postconditions

success_rate

Molt 触发（经验反馈）：

基于成功/失败执行记录自动调整成功率

Agent 可选择更稳定的流程

🔁 2. Molt 元进化控制器（Memory Meta-Controller）

这是整个系统最核心的升级：

它不是一个单一数据库，而是一个负责：

周期性评估记忆质量（召回命中、任务成功率）

调整 Memory 系统行为（升降级规则、Prompt 变更、索引策略）

自动优化淘汰与保留参数

推动长期结构自适应进化

🧠 Molt 循环机制
记忆生成 → 质量评估 → 反馈信号 → 调整策略 → 更新记忆规则

2.1 质量评估信号来源

Agent 的任务成功 / 失败

记忆召回命中率统计

复习强度 / 任务相关度

冲突信息率

🪄 自我修改策略（自动）

Molt 控制器可以自动：

调整向量索引参数（如 recall threshold）

修改分块与摘要 Prompt 产生规则

自动压缩冗余记忆

调整淘汰阈值

重新组织图关系

这样的机制与研究中强调的 “记忆的动态更新 & 衰退控制” 相呼应，而不是静态的记忆池。

🔐 3. 多 Agent 共享 / 竞争记忆规范
层级	共享	私有
M1	❌	✔️
M2	❌	✔️
M3	⚠️ 授权由 Molt Collector 控制	✔️
LTM	✔️（版本 & 来源可追踪）	✔️
Procedural	✔️（安全执行 & 审计）	✔️
⛔ 共享层级禁止行为

私有经验直接污染 LTM

无共识写入长期知识

Agent 直接操作底层结构

🧪 4. Molt 驱动的验证与测试体系

为了避免“黑箱成长”，每一层必须：

能回答：我是哪个层级记忆

能回答：我什么时候会被遗忘

能回答：我从哪些经验中抽象而来

可审计来源与时间线

不允许 Agent 直接写底层 schema

🧾 5. 最终验收 Checklist（加 Molt 指标）

❏ M3 到 LTM 的抽象是否触发正确

❏ Molt 是否主动调整 Prompt / 压缩策略

❏ 冲突知识是否保留来源版本

❏ 所有记忆可解释 & 可追踪 & 可测试

❏ 多 Agent 读写共享规范生效

🧠 总体架构图（逻辑）
             ┌──────────────┐
             │  Input Stream│
             └───────▲──────┘
                     │
M1 Sensory Cache ───▶│
                     │        Molt Meta Controller
                     ▼      ↙        ▲      ↘
             ┌──────────────┐       │       ┌──────────────┐
             │   M2 WM      │◀──Quality Feedback──▶ M3 Episodic Memory
             └──────────────┘       │       └──────────────┘
                     │              │                │
                     │              ▼                ▼
                     └─────────▶ LTM   ←────────▶ Procedural
                                (abstract / shared knowledge)