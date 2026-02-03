# 长期记忆实现规范（统一版本）

基于当前技术栈：MongoDB + Milvus + Elasticsearch + Neo4j，并补充三个关键能力：

- 抽象（Abstraction）
- 反馈机制（Feedback）
- 记忆衰减与时间维度、版本迭代（Decay / Versions）

本规范统一描述系统整体架构、记忆分层、反馈信号、演化调度与版本控制，不再区分版本。

## 1. 系统总体架构

User Input
   ↓
Agent Core（多级推理 + 输出生成）
   ↓
Response
   ↓
Feedback Judge（独立模型/模块）
   ↓
Evolution Dispatcher
   ↓
Memory / Skill / Strategy Callback

核心原则

- 职责分离
- 非侵入式调整
- 多级记忆对齐

职责说明

- Agent Core：推理、生成、执行任务
- Feedback Judge：评估用户反馈、生成信号
- Evolution Dispatcher：决定记忆或技能调整
- Memory/Skill：被动接收、逐步调整

非侵入式调整

- 单轮信号不直接修改长期记忆
- 调整依赖多轮一致信号或高置信度

多级记忆对齐

- M1：感觉缓存（本轮短期状态）
- M2：工作记忆（短期、临时权重调整）
- M3：稳定记忆（长期、需多源验证）

## 2. 分层记忆实现

### M1 感觉缓存（Sensory）

职责：只存原始输入流
组件：MongoDB + TTL

实现细节

- 存储：MongoDB collection（感知流）
- TTL index 让数据自动过期
- 触发规则（供调度器读取）
- 高频触达可标记 candidate 上升
- 不向量化、不参与高层检索

### M2 工作记忆（Working Memory）

职责：当前任务临时上下文
实现：内存结构（进程运行）

规则

- fixed cap（如 7±2）
- rehearsal count 影响淘汰顺序
- 不落库、不跨会话、不共享

### M3 短期记忆 / 经验记忆（Episodic）

核心存储层，包含向量、图结构和元信息。

组件组合

- Milvus：语义向量召回
- Neo4j：情境关系图
- MongoDB：原始片段与元信息

优势

- 语义检索与关系推理可并存

新增能力：反馈与衰减

反馈机制

- 每次记忆被检索或任务成功使用
  - importance_score += 1
  - last_reviewed = now
- 未使用的记忆
  - importance_score *= decay_factor

衰减机制

每条记忆记录包含

- created_at
- importance_score
- last_reviewed
- decay_rate

衰减规则

importance_score := importance_score * exp(-lambda * time_since_last_review)

时间维度

- MongoDB 存储时间戳
- 可按时间筛选过时记忆

### 知识图关系（Neo4j）

节点关系

(M3_Item)-[:RELATED_TO {type, weight}]->(Other_Item)

权重调整

- 正向引用增加权重
- 过时或不相关逐步衰减

### Long-Term Memory（LTM）

LTM 的核心实体

- 抽象结论
- 条件
- 置信度
- 来源关系

抽象定义

- 从若干 M3 片段合成高阶结论

技术实现

- Neo4j：抽象结论节点
- Milvus：结论向量
- ES：关键词 Tag

差异点

- LTM 不存原始长文本
- 只存结论、条件、来源

版本与时间

LTM 节点字段

- version
- timestamp
- sources[]
- confidence

冲突策略

- 同源合并
- 异源冲突保留版本

## 3. 抽象引擎（Abstraction Engine）

功能

- 自动发现主题相似的 M3 片段
- 合并为 condensed summary
- 标记适用条件与置信度
- 输出标准化结论

实现方式

- LLM 或模板逻辑抽象
- 输出写入 Neo4j LTM
- 生成向量写入 Milvus

## 4. Feedback Judge

职责

- 独立模型或模块
- 只分析用户输入 + Agent 输出 + 上下文短窗口
- 输出信号，不直接修改记忆

输入

```
{
  "user_input": "...",
  "agent_response": "...",
  "context_window": ["prev_user", "prev_agent", ...]
}
```

输出信号

```
{
  "feedback_detected": true,
  "type": "negative | positive | neutral | mixed",
  "confidence": 0.0-1.0,
  "targets": ["reasoning", "final_answer", "style", "tool_choice"],
  "action_hint": ["penalty", "invalidate", "prefer"]
}
```

特性

- 不参与对话生成
- 可异步处理
- 可独立升级模型
- 信号触发 Dispatcher，但不直接改记忆

## 5. Evolution Dispatcher

功能

- 接收 Judge 输出信号
- 决定触发的回调（Memory/Skill 调整）
- 控制调整时机与粒度
- 支持多轮信号聚合

核心逻辑

```
for signal in signal_pool:
    if signal.confidence < threshold:
        continue
    if signal.type == 'negative':
        emit_penalty(signal.targets)
    elif signal.type == 'mixed':
        split_targets(signal.targets)
apply_aggregated_adjustments()
```

## 6. Memory / Skill 调整策略

层级与触发

- M1 感觉缓存：本轮失败标记，Judge 信号即时反馈
- M2 工作记忆：临时权重调整/技能偏好，Judge 多轮确认 + 高置信度
- M3 稳定记忆：长期知识/技能调整，多源一致 + 高置信度 + 演化策略同意

特性

- 调整渐进、可回滚
- 支持多信号累积
- 可针对 reasoning / style / tool_choice 精细调整
- 记录调整历史，便于审计

## 7. 版本控制与冲突管理

需求

- 新证据 vs 旧证据
- 冲突结论并存

版本策略

- 源相同 → 合并
- 源冲突不同 → 保留版本

## 8. 接口规范

Agent Core → Judge

```
judge_input = {
    "user_input": user_text,
    "agent_response": agent_output,
    "context_window": context_window
}
feedback_signal = judge.evaluate(judge_input)
```

Judge → Dispatcher

```
dispatcher.enqueue(feedback_signal)
dispatcher.aggregate_and_apply()
```

Dispatcher → Memory/Skill

```
memory.apply_adjustments(aggregated_signals)
skill_manager.adjust_weights(aggregated_signals)
```

## 9. 衰减与生命周期

衰减公式

importance = importance * exp(-λ * Δt)

说明

- λ 可按层级与使用频率动态调整
- 衰减越久权重越低，最终可淘汰

## 10. Molt Controller（Meta-Feedback）

职责

- 调整 recall threshold
- 调整 decay rate
- 触发抽象引擎
- 冲突处理策略

触发条件示例

| 动作 | 触发条件 |
|---|---|
| 调整 recall threshold | 召回效果下降 |
| 调整 decay rate | 记忆过早衰减或混乱 |
| 触发抽象引擎 | 相关片段密集 |
| 冲突处理策略 | LTM 冲突高 |

调度器需要定期评估各层反馈统计指标并动态调整策略。

## 11. 最终验收标准（可测试）

检查记忆生命周期

| 检查项 | 预期行为 |
|---|---|
| Episodic importance 衰减 | 随时间自动下降 |
| 高频复习记忆强化 | 权重上升 |
| 抽象节点生成 | M3 相关片段密集时触发 |
| 冲突版本保留 | 源不同但同主题 |
| Molt 策略调整 | recall 成效指标改善 |

可解释性

- 为什么该记忆被保留
- 来源事件
- 过期与强化时间

## 12. 备选与扩展

时间数据库替代

- MongoDB TTL 不够精细时，可考虑 PostgreSQL + TimescaleDB

总结要点

- 记忆层不仅存储，还要有抽象、反馈、衰减
- 时间维度与版本迭代缺一不可
