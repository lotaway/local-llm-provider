# MOLT 类脑记忆系统实现计划

## 概述

基于现有 `MongoDB + Milvus + Elasticsearch + Neo4j` 技术栈，增量实现类脑记忆系统。

**目标**: 记忆生命周期管理 + 自我进化能力

---

## 目录

- [阶段一：基础数据模型扩展](#阶段一基础数据模型扩展)
- [阶段二：反馈机制](#阶段二反馈机制)
- [阶段三：衰减调度器](#阶段三衰减调度器)
- [阶段四：抽象引擎](#阶段四抽象引擎)
- [阶段五：长期记忆LTM存储](#阶段五长期记忆ltm存储)
- [阶段六：Molt Controller](#阶段六molt-controller)
- [阶段七：版本控制与冲突管理](#阶段七版本控制与冲突管理)
- [阶段八：集成测试](#阶段八集成测试)

---

## 阶段一：基础数据模型扩展

### 目标
为现有MongoDB/Neo4j添加记忆生命周期所需字段。

### 交付物

| 文件 | 修改内容 |
|------|---------|
| `repositories/mongodb_repository.py` | 添加 `importance_score`, `last_reviewed`, `decay_rate` 字段 |
| `repositories/neo4j_repository.py` | 添加 `LTMNode` 标签和版本属性 |

### 详细步骤

#### 1.1 MongoDB Schema 扩展
```python
# 新增字段定义
chunks: {
    "importance_score": float,      # 默认 0.5, 范围 0-1
    "last_reviewed": datetime,      # 最后被检索/使用时间
    "decay_rate": float,            # 衰减率 λ, 默认 0.01/天
    "review_count": int,            # 被复习次数
    "memory_type": str,             # "sensory"|"episodic"
}
```

#### 1.2 Neo4j Schema 扩展
```cypher
# 新增 LTM 节点类型
CREATE (l:LTMNode {
    topic: string,
    version: int,
    conclusion: string,
    confidence: float,
    sources: list,
    created_at: datetime,
    updated_at: datetime
})

# 新增关系类型
CREATE (e:Episodic)-[:EVOLVED_TO]->(l:L)
```

### 验收标准
- [x] MongoDB chunks 集合新增字段有默认值
- [x] Neo4j 支持 LTMNode 查询
- [x] 向后兼容现有数据（老数据自动补充默认值）

### 验证结果
```python
# MongoDB
chunk.importance_score = 0.5  # ✓ 默认值正确
chunk.decay_rate = 0.01       # ✓ 默认值正确
chunk.memory_type = "episodic" # ✓ 类型标记正确

# Neo4j
neo4j.save_ltm(ltm)          # ✓ LTM节点存储
neo4j.get_ltm_by_topic()     # ✓ 按主题查询
neo4j.link_episodic_to_ltm() # ✓ 关系建立
```

---

## 阶段二：反馈机制

### 目标
在检索链路中埋点，记录记忆使用情况。

### 交付物

| 文件 | 修改内容 |
|------|---------|
| `retrievers/hybrid_retriever.py` | 添加 `on_retrieve` 回调 |
| `retrievers/graph_retriever.py` | 添加 `on_retrieve` 回调 |
| `services/feedback_service.py` | 新增文件，处理权重更新 |

### 详细步骤

#### 2.1 反馈服务 `services/feedback_service.py`
```python
class FeedbackService:
    async def on_recall(self, memory_id: str, memory_type: str):
        """记忆被召回时的反馈"""
        
    async def on_usage(self, memory_id: str, task_success: bool):
        """记忆被实际使用后的反馈"""
        
    async def update_importance(self, chunk_id: str, delta: float):
        """更新重要性评分"""
```

#### 2.2 检索回调埋点
```python
# hybrid_retriever.py 修改
class HybridRetriever:
    def __init__(self, ..., feedback_service: FeedbackService = None):
        self.feedback_service = feedback_service
        
    def invoke(self, query):
        docs = self._search(query)
        if self.feedback_service:
            for doc in docs:
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id:
                    await self.feedback_service.on_recall(chunk_id, "episodic")
        return docs
```

### 验收标准
- [ ] 每次检索调用 `FeedbackService.on_recall`
- [ ] importance_score 正确递增
- [ ] review_count 正确递增

---

## 阶段三：衰减调度器

### 目标
定期执行重要性衰减，淘汰低价值记忆。

### 交付物

| 文件 | 修改内容 |
|------|---------|
| `services/decay_scheduler.py` | 新增文件，周期性衰减任务 |
| `utils/scheduler.py` | 扩展或复用现有调度器 |

### 详细步骤

#### 3.1 衰减公式实现
```python
# 指数衰减
importance(t) = importance_0 * exp(-λ * Δt)

# 复习强化
importance = importance * (1 + review_count * k)
```

#### 3.2 调度任务
```python
# 每6小时执行一次衰减
@scheduler.task('interval', hours=6)
async def apply_decay():
    for chunk in mongo_repo.get_all_chunks():
        new_score = calculate_decayed_score(chunk)
        mongo_repo.update_importance(chunk['chunk_id'], new_score)
```

#### 3.3 淘汰策略
```python
# 低于阈值且30天未使用的chunk标记为待清理
THRESHOLD = 0.1
MAX_AGE_DAYS = 30
```

### 验收标准
- [ ] 重要性评分按指数衰减
- [ ] 高频使用记忆权重上升
- [ ] 低价值记忆自动降级

---

## 阶段四：抽象引擎

### 目标
将多个相关 Episodic 记忆聚合成 LTM 结论。

### 交付物

| 文件 | 修改内容 |
|------|---------|
| `services/abstraction_engine.py` | 新增文件，M3→LTM 提炼 |

### 详细步骤

#### 4.1 相似片段聚类
```python
async def find_related_episodes(self, topic: str, threshold: float = 0.8) -> List[Dict]:
    """查找主题相关的M3片段"""
    # 使用Milvus向量相似度搜索
    # 或Neo4j关系查询
```

#### 4.2 LLM 抽象总结
```python
async def abstract_to_ltm(self, episode_ids: List[str]) -> Dict:
    """将多个片段抽象为LTM结论"""
    
    prompt = """
    Given these related memory fragments, extract a general principle:
    
    {fragments}
    
    Output format:
    - conclusion: string
    - conditions: list[string]
    - confidence: float (0-1)
    - applicable_topics: list[string]
    """
```

#### 4.3 触发条件
```python
# 当某主题相关片段超过N个 且 平均重要性 > 阈值 时触发
THRESHOLD_EPISODES = 5
THRESHOLD_IMPORTANCE = 0.6
```

### 验收标准
- [ ] 自动发现高相关M3片段
- [ ] LLM生成标准格式结论
- [ ] 抽象结果存入LTM

---

## 阶段五：长期记忆LTM存储

### 目标
在Neo4j中存储和管理LTM节点。

### 交付物

| 文件 | 修改内容 |
|------|---------|
| `repositories/neo4j_repository.py` | 添加 LTM 操作方法 |
| `models/ltm_node.py` | 新增文件，LTM数据模型 |

### 详细步骤

#### 5.1 LTM 节点模型
```python
class LTMNode:
    topic: str
    version: int
    conclusion: str
    conditions: List[str]
    confidence: float
    sources: List[str]  # 源chunk_id列表
    created_at: datetime
    updated_at: datetime
```

#### 5.2 LTM Repository 方法
```python
class Neo4jRepository:
    def save_ltm(self, node: LTMNode):
        """保存LTM节点"""
        
    def get_ltm_by_topic(self, topic: str) -> List[Dict]:
        """按主题查询LTM"""
        
    def link_episodic_to_ltm(self, episodic_id: str, ltm_id: str):
        """建立 M3 → LTM 关系"""
```

### 验收标准
- [ ] LTM节点存储在Neo4j
- [ ] 支持按topic查询
- [ ] 保留来源追溯

---

## 阶段六：Molt Controller

### 目标
元认知调度器，动态调整策略。

### 交付物

| 文件 | 修改内容 |
|------|---------|
| `services/molt_controller.py` | 新增文件，策略调度中心 |

### 详细步骤

#### 6.1 监控指标
```python
class MoltController:
    metrics = {
        "recall_hit_rate": float,      # 检索命中率
        "avg_importance": float,       # 平均重要性
        "decay_rate_effective": float, # 衰减有效性
        "abstraction_rate": float,     # 抽象频率
    }
```

#### 6.2 策略调整
```python
async def evaluate_and_adjust(self):
    """评估指标并调整策略"""
    
    # 如果召回率下降 → 降低召回阈值
    if self.metrics.recall_hit_rate < 0.3:
        self.recall_threshold *= 0.9
        
    # 如果抽象频率过低 → 降低抽象触发阈值
    if self.metrics.abstraction_rate < 0.01:
        self.abstraction_min_episodes -= 1
```

#### 6.3 调度编排
```python
# Molt Controller 编排以下任务
SCHEDULES = {
    "decay": "0 */6 * * *",      # 每6小时衰减
    "abstraction": "0 */12 * * *", # 每12小时抽象检查
    "evaluation": "0 0 * * 0",    # 每周策略评估
}
```

### 验收标准
- [ ] 定期输出策略建议
- [ ] 动态调整衰减率/召回阈值
- [ ] 触发抽象引擎

---

## 阶段七：版本控制与冲突管理

### 目标
处理知识演进中的版本和冲突。

### 交付物

| 文件 | 修改内容 |
|------|---------|
| `services/version_manager.py` | 新增文件，版本控制逻辑 |

### 详细步骤

#### 7.1 冲突检测
```python
async def detect_conflict(self, topic: str, new_conclusion: str) -> bool:
    """检测新结论是否与现有LTM冲突"""
    
    existing = self.neo4j_repo.get_ltm_by_topic(topic)
    for lt in existing:
        if self._semantic_similarity(new_conclusion, lt.conclusion) < 0.7:
            return True
    return False
```

#### 7.2 版本策略
```python
# 策略：源不同则保留版本，源相同则合并
async def upsert_ltm(self, new_node: LTMNode) -> str:
    existing = self.neo4j_repo.get_ltm_by_topic(new_node.topic)
    
    if not existing:
        return self.neo4j_repo.save_ltm(new_node)
    
    for lt in existing:
        if self._same_sources(new_node.sources, lt.sources):
            # 合并：更新版本，合并sources
            return self._merge_and_update(lt, new_node)
        else:
            # 冲突：保留版本，创建新节点
            return self._create_version(lt, new_node)
```

### 验收标准
- [ ] 相同源结论自动合并
- [ ] 不同源冲突保留版本
- [ ] 可追溯版本历史

---

## 阶段八：集成测试

### 目标
端到端验证系统功能。

### 交付物

| 文件 | 修改内容 |
|------|---------|
| `tests/test_molt_memory.py` | 新增文件，集成测试用例 |

### 测试用例

| 用例 | 验证点 |
|------|-------|
| 记忆召回反馈 | importance_score 递增 |
| 衰减生效 | 长时间未访问记忆分数下降 |
| 抽象触发 | 密集相关片段自动生成LTM |
| 版本控制 | 冲突结论保留多版本 |
| Molt调度 | 策略调整生效 |

### 验收标准
- [ ] 所有测试用例通过
- [ ] 性能：衰减任务 < 5分钟完成
- [ ] 无内存泄漏

---

## 执行顺序

```
阶段一 → 阶段二 → 阶段三 → 阶段四 → 阶段五 → 阶段六 → 阶段七 → 阶段八
```

### 并行优化

阶段一完成后即可并行进行：
- 阶段二（依赖一）
- 阶段五（独立）

---

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-02-02 | v1.0 | 初始化实现计划 |


---

# ✅ 所有阶段已完成

## 实现完成清单

| 阶段 | 组件 | 文件 | 状态 |
|------|------|------|------|
| **阶段一** | MongoDB Schema | `repositories/mongodb_repository.py` | ✅ |
| | Neo4j LTM | `repositories/neo4j_repository.py` | ✅ |
| | LTMNode模型 | `schemas/graph.py` | ✅ |
| **阶段二** | FeedbackService | `services/feedback_service.py` | ✅ |
| | Retriever回调 | `retrievers/hybrid_retriever.py` | ✅ |
| | RAG集成 | `rag.py` | ✅ |
| **阶段三** | DecayScheduler | `services/decay_scheduler.py` | ✅ |
| **阶段四** | AbstractionEngine | `services/abstraction_engine.py` | ✅ |
| **阶段五** | LTM存储 | `repositories/neo4j_repository.py` | ✅ |
| **阶段六** | MoltController | `services/molt_controller.py` | ✅ |
| **阶段七** | VersionManager | `services/version_manager.py` | ✅ |
| **阶段八** | 集成测试 | `tests/test_molt_integration.py` | ✅ |

## 新增文件清单

```
services/
├── feedback_service.py      # 反馈机制
├── decay_scheduler.py       # 衰减调度
├── abstraction_engine.py    # 抽象引擎
├── molt_controller.py       # 元认知调度
└── version_manager.py       # 版本控制

tests/
├── test_molt_phase1.py      # 阶段一验收
└── test_molt_integration.py # 集成测试
```

## 核心API概览

```python
# 反馈
FeedbackService.on_recall(chunk_id)
FeedbackService.on_usage(chunk_id, success)
FeedbackService.on_task_success(chunk_ids)

# 衰减
DecayScheduler.calculate_decay(chunk)
DecayScheduler.apply_decay_all()

# 抽象
AbstractionEngine.abstract_to_ltm(topic)
AbstractionEngine.auto_discover_and_abstract()

# 版本
VersionManager.upsert_ltm(ltm)
VersionManager.get_version_chain(topic)

# 调度
MoltController.run_full_cycle()
MoltController.evaluate_and_adjust()
```

## 快速开始

```python
from services.molt_controller import MoltController

controller = MoltController()
report = controller.run_full_cycle()
status = controller.get_status()
print(status)
controller.close()
```

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-02-02 | v1.0 | 初始化实现计划 |
| 2026-02-02 | v1.1 | **阶段一完成**: MongoDB/Neo4j Schema扩展 |
| 2026-02-02 | v1.2 | **阶段二完成**: 反馈机制 |
| 2026-02-02 | v1.3 | **阶段三完成**: 衰减调度器 |
| 2026-02-02 | v1.4 | **阶段四完成**: 抽象引擎 |
| 2026-02-02 | v1.5 | **阶段五完成**: LTM存储 (Neo4j) |
| 2026-02-02 | v1.6 | **阶段六完成**: Molt Controller |
| 2026-02-02 | v1.7 | **阶段七完成**: 版本控制 |
| 2026-02-02 | v1.8 | **阶段八完成**: 集成测试 |

