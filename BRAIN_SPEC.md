类脑多级记忆系统（Agent 用）实现规范
0. 总体目标（Agent 视角）

Agent 的记忆系统必须满足以下目标：

区分记忆层级，而不是所有内容一股脑丢进 RAG

允许遗忘，且遗忘是可控、可解释的

支持多 Agent 的共享 / 私有 / 共识记忆

长期记忆必须被压缩、抽象、索引，而非全文堆积

所有记忆必须“可被检索、可被解释、可被测试”

禁止出现：

无限增长的上下文

无生命周期的“永久记忆”

Agent 直接操作数据库细节

1. 记忆层级定义（强约束）
M1：感觉缓存（Sensory / Immediate）

目标

承载“当前输入流”

不参与知识积累

不允许被长期引用

重点

TTL

注意力筛选

零结构、零抽象

实现方式

MongoDB + TTL Index

字段：

timestamp

source

attention_score

raw_content

禁忌

❌ 不允许向量化

❌ 不允许进入 Neo4j

❌ 不允许被 Agent 主动查询

测试用例

写入后 TTL 到期是否自动消失

attention_score < threshold 是否不会进入 M2

验收方式

M1 中任何数据在 TTL 后不可检索

Agent 无法通过任何接口主动回忆 M1

质量标准

延迟 < 内存级

零持久语义污染

M2：工作记忆（Working Memory）

目标

支持当前推理、规划、决策

严格容量限制

重点

Chunk 化

复述计数

容量淘汰策略

实现方式

Agent 进程内结构（非数据库）

结构字段：

content

created_at

last_accessed

rehearsal_count

links_to_other_chunks

禁忌

❌ 不允许直接落库

❌ 不允许被其他 Agent 直接读取

❌ 不允许超过容量上限（必须丢）

测试用例

超过容量时是否正确淘汰低 rehearsal 项

rehearsal 是否延缓淘汰

验收方式

任一 Agent 重启后 M2 必须清空

同一任务内上下文连贯但任务结束即消失

质量标准

行为符合“7±2 组块”

无跨任务污染

M3：短期记忆（Short-term / Episodic）

目标

可被回忆

可被遗忘

可被强化

重点

向量 + 图双存储

艾宾浩斯遗忘调度

情境记忆

实现方式

Milvus / ES：向量

Neo4j：情境关系

MongoDB：原始片段（非全文索引）

最小字段

embedding

semantic_summary

created_at

last_reviewed

importance

graph_links

禁忌

❌ 不允许全文无脑进 ES

❌ 不允许无 review 记录

❌ 不允许直接作为“事实真理”

测试用例

未复习记忆是否随时间权重下降

多次复习是否推迟遗忘

图关系是否可反向追溯上下文

验收方式

同一问题，近期记忆优先命中

多次引用后权重明显提升

质量标准

Recall 可解释

权重变化可追踪

2. 长期记忆（Long-term Memory, LTM）
目标

存“抽象后的知识”

去除时间噪声

高复用、低冗余

重点

压缩

抽象

多索引

实现方式
压缩规则（必须）

多条相似 M3 → 一个 LTM 节点

不保留原始对话

只保留：

抽象结论

适用条件

置信度

存储分工

Neo4j：概念、因果、依赖

Milvus：语义召回

ES：仅用于 tag / keyword（非全文）

禁忌

❌ 不允许存时间序列对话

❌ 不允许直接写事实而无来源

❌ 不允许 LTM 无图关系

测试用例

删除 M3 后 LTM 是否仍可用

相同事实多次学习是否被合并

冲突知识是否可并存（带置信度）

验收方式

LTM 节点平均连接度 > 1

无孤立长期节点

质量标准

信息密度高

冗余率低

可被多 Agent 复用

3. 程序性记忆（Procedural Memory）
目标

存“怎么做”

而不是“做了什么”

重点

稳定流程

成功率反馈

可复用调用

实现方式

Neo4j Skill Graph

每个节点包含：

step_pattern

preconditions

postconditions

success_rate

禁忌

❌ 不允许自然语言步骤堆叠

❌ 不允许不可验证流程

❌ 不允许无执行反馈

测试用例

多次执行是否提高 success_rate

失败是否自动降权

Agent 是否能选择成功率更高的流程

验收方式

同任务多次执行成功率上升

Agent 能解释为何选择某流程

质量标准

稳定

可组合

可演化

4. 多 Agent 共享记忆规范
共享层级（硬规则）
层级	是否共享
M1	❌
M2	❌
M3	⚠️ 需授权
LTM	✅
Procedural	✅
重点

权限标签

版本控制

冲突合并

禁忌

❌ Agent 私有经验直接污染 LTM

❌ 无共识直接写共享知识

测试用例

Agent A 写入后 Agent B 是否可控读取

冲突事实是否保留多版本

验收方式

无“单 Agent 霸权记忆”

冲突可追溯来源

5. 记忆系统全局禁忌（非常重要）

❌ 把 ES / Mongo 当万能全文存储
❌ 用“召回率”掩盖结构缺失
❌ 没有遗忘机制
❌ Agent 直接写数据库 schema
❌ 把上下文当长期记忆

6. 最终验收 Checklist（给你自己用）

 任一记忆都能回答：我属于哪一层

 任一记忆都能回答：我什么时候会被忘

 任一长期知识都能回答：我从哪里抽象而来

 任一 Agent 都不能拥有“不可审计的记忆”

 GraphRAG 不再是“唯一大脑”，而是记忆的一层