Milvus + Neo4j Graph RAG 智能体设计规范
目标（Goals）

设计并实现一个基于 Neo4j（结构化知识图） + Milvus（向量检索） 的智能体，用于：

支持跨文档、跨实体的结构化推理

支持增量学习（新增文档不破坏既有知识）

为 Agent 提供可解释的检索 + 推理上下文

同时支持自然语言查询与工具/规划型调用

非目标（Non-Goals）

通用搜索引擎：不追求全文搜索或关键词排名

事实裁决系统：不保证“唯一正确答案”

强一致知识库：不做强约束一致性推理（如 OWL reasoner）

在线训练模型：不在系统内做模型微调

系统约束（Constraints）
存储与职责划分

Neo4j

存储实体、关系、证据元数据

负责结构遍历、路径扩展、关系约束

Milvus

存储文本、实体、关系的 embedding

负责相似度检索、候选召回

LLM

不直接访问全量图

只对“子图 + 文本证据”做推理

执行模型

单 Agent 主循环（Single-agent, single-thread logical flow）

支持工具调用（Graph 查询 / Vector 查询）

所有写操作必须是幂等的

核心概念定义（Definitions）
Entity（实体）

系统中一切可被引用、推理、连接的对象，例如：

Document

CodeArtifact

API

Concept

Model

Event

每个实体必须包含：

stable_id

type

canonical_name

created_at

Relation（关系）

描述实体之间的语义连接，例如：

uses

depends_on

implements

contradicts

refines

references

每条关系必须包含：

from_entity_id

to_entity_id

relation_type

confidence

source_doc_id

Evidence（证据）

支持某实体或关系的最小文本单元：

文档段落

代码片段

注释

论文结论句

Evidence 只存在于 Milvus 中，Neo4j 仅存引用。

系统不变式（Invariants）
图结构不变式

实体唯一性：同一 stable_id 只能对应一个实体节点

关系不可覆盖：已存在关系不得被删除或语义替换

时间可追溯：所有实体和关系必须保留 source 与 timestamp

向量一致性不变式

任意 Neo4j 实体 / 关系必须：

要么有对应 embedding

要么被显式标记为 non-embeddable

不允许“孤立向量”无图引用

数据流设计（Data Flow）
文档增量写入流程
New Document
  ↓
Chunking & Metadata
  ↓
Entity / Relation Extraction
  ↓
Entity Resolution（对齐旧图）
  ↓
Neo4j: Node / Edge Merge
  ↓
Milvus: Evidence / Node Embedding Insert

查询推理流程
User Query
  ↓
Milvus 相似度召回（Evidence / Entity）
  ↓
Neo4j 子图扩展（N-hop）
  ↓
上下文裁剪（Token Budget）
  ↓
LLM 推理 / 决策

智能体能力边界（Agent Capabilities）
Agent 可以做的

查询 Milvus 获取候选知识

基于实体 ID 查询 Neo4j 关系

在子图内做多跳推理

输出结构化中间结果（不是只给文本）

Agent 不可以做的

扫描全图

自行修改 Schema

删除既有实体或关系

推断“未显式建模的事实”为确定事实

Schema 演进规则（关键）
允许的演进

新增实体子类型（必须声明父类）

新增关系类型（不得与既有语义冲突）

增加属性字段

禁止的演进

重命名既有实体 / 关系类型

改变关系方向含义

删除历史关系

失败模式与处理（Failure Modes）
向量检索失败

回退到结构化图遍历

明确标注“低置信度回答”

图冲突（语义矛盾）

保留全部冲突关系

不在存储层裁决

由 LLM 在推理阶段基于证据权重判断

验收标准（Acceptance Criteria）
基础能力

连续写入 ≥ 1000 篇文档，不发生实体重复爆炸

新文档写入后，不影响既有查询结果

推理能力

能回答至少 3-hop 的跨文档问题

回答中能明确引用实体 / 关系来源

稳定性

任意一次 Agent 推理，不得访问全量图

Neo4j / Milvus 任一不可用时，系统可降级运行