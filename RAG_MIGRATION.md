# RAG 系统架构迁移指南

## 概述

本次重构将原本独立使用的 Milvus、Elasticsearch、Neo4j 整合为统一架构，以 **MongoDB 作为事实源（Source of Truth）**。

## 架构变更

### 迁移前（独立模式）

```
┌─────────┐     ┌─────────────────┐     ┌─────────┐
│ Milvus  │     │ Elasticsearch   │     │ Neo4j   │
│(向量库) │     │   (全文检索)    │     │(图数据库)│
└─────────┘     └─────────────────┘     └─────────┘
     ↑                  ↑                    ↑
     │                  │                    │
     └──────────────────┼────────────────────┘
                        │
                   独立写入，无关联
```

### 迁移后（统一架构）

```
                    ┌─────────────┐
                    │   MongoDB   │
                    │ (事实源)    │
                    │ documents   │
                    │ chunks      │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Milvus    │ │Elasticsearch│ │   Neo4j     │
    │ chunk向量   │ │  全文索引   │ │ 结构索引    │
    │ chunk文本   │ │  chunk文本  │ │ (仅存ID)    │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                    通过 chunk_id / doc_id 关联
```

## 核心变更

### 1. MongoDB Repository (`repositories/mongodb_repository.py`)

新增 MongoDB 仓库，管理：
- `documents` 集合：存储原始文档
- `chunks` 集合：存储分块数据

```python
from repositories.mongodb_repository import MongoDBRepository

# 文档结构
{
    "_id": "doc_xxx",
    "source": "filesystem",
    "path": "/path/to/file",
    "filename": "file.md",
    "content": "完整文档内容",
    "checksum": "sha256:...",
    "status": {
        "chunked": True,
        "embedded": True,
        "graphed": True
    }
}

# Chunk 结构
{
    "chunk_id": "doc_xxx_chunk_0001",
    "doc_id": "doc_xxx",
    "index": 1,
    "text": "chunk 文本",
    "embedding_status": "done"
}
```

### 2. LocalRAG 类变更

#### 初始化
```python
# 新增 MongoDB 初始化
self.mongo_repo = MongoDBRepository()

# Milvus collection 名称变更
self.collection = DB_COLLECTION
```

#### 文档处理流程
```python
async def add_document(self, title, content, source, ...):
    # 1. 计算 checksum，生成 doc_id
    # 2. 保存文档到 MongoDB
    # 3. 生成 chunks，保存到 MongoDB（带 chunk_id）
    # 4. 向量化保存到 Milvus
    # 5. 索引到 Elasticsearch
    # 6. 抽取图结构到 Neo4j（仅 chunk_id 引用）
```

### 3. Neo4j 节点设计

```cypher
-- Document 节点
(:Document {doc_id, filename})

-- Chunk 指针节点（不存文本/向量）
(:Chunk {chunk_id, doc_id, index})

-- Concept 节点
(:Concept {stable_id, name})

-- 关系
(Document)-[:CONTAINS]->(Chunk)
(Chunk)-[:DESCRIBES]->(Concept)
(Concept)-[:DEPENDS_ON]->(Concept)
```

### 4. 新增辅助方法

```python
# 从 MongoDB 获取完整文档
def get_document_context(self, doc_id: str) -> str

# 从 MongoDB 获取 chunk
def get_chunk_context(self, chunk_id: str) -> str

# 获取上下文 chunks
def get_surrounding_chunks(self, chunk_id: str, before=1, after=1) -> List[str]

# 清理连接
def close(self)
```

## 环境变量

在 `.env` 或 `.env.example` 中添加：

```bash
# MongoDB (Source of Truth)
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=rag_system

# Milvus (Vector Storage)
DB_HOST=localhost
DB_PORT=19530
DB_COLLECTION=document_chunks

# Elasticsearch (Full-text Search)
ES_HOST=localhost
ES_PORT1=9200
ES_INDEX_NAME=rag_docs

# Neo4j (Graph Structure)
NEO4J_BOLT_URL=bolt://localhost:7687
NEO4J_AUTH=neo4j/password
```

## Docker Compose

已在 `docker-compose.env.yml` 中添加 MongoDB 服务：

```yaml
mongodb:
  image: mongo:8.0
  container_name: llp-mongodb
  environment:
    MONGO_INITDB_ROOT_USERNAME: ${MONGO_ROOT_USER:-admin}
    MONGO_INITDB_ROOT_PASSWORD: ${MONGO_ROOT_PASSWORD:-123123123}
  ports:
    - "${MONGO_PORT:-27017}:27017"
  volumes:
    - mongodb_data:/data/db
```

启动所有服务：
```bash
docker-compose -f docker-compose.env.yml up -d
```

## 查询协同示例

问题："Attention 在 Transformer 中如何实现"

```python
# 1. Neo4j：查找相关 Concept
concepts = neo4j.query("MATCH (c:Concept) WHERE c.name CONTAINS 'Attention' RETURN c")

# 2. Neo4j：获取关联的 chunk_id
chunk_ids = neo4j.query("""
    MATCH (c:Concept {name: 'Attention'})<-[:DESCRIBES]-(chunk:Chunk)
    RETURN chunk.chunk_id
""")

# 3. Milvus：按 chunk_id 过滤 + 向量相似度
results = milvus.search(query_vector, filter=f"chunk_id in {chunk_ids}")

# 4. ES：补充关键词结果
es_results = es.search(query="Attention Transformer")

# 5. MongoDB：如需完整上下文，回溯原文档
doc = mongo.get_document(doc_id)
```

## 迁移步骤

1. **启动 MongoDB**
   ```bash
   docker-compose -f docker-compose.env.yml up -d mongodb
   ```

2. **更新环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 添加 MONGO_URI 等配置
   ```

3. **安装依赖**
   ```bash
   pip install pymongo
   ```

4. **重新索引文档**（可选）
   如需将现有数据迁移到新架构，运行：
   ```bash
   python scripts/cleanup_rag_data.py --all
   # 然后重新导入文档
   ```

## 注意事项

1. **ID 稳定性**：`doc_id` 基于内容 SHA256 前16位，`chunk_id` 格式为 `{doc_id}_chunk_{index:04d}`
2. **幂等性**：重复导入相同内容会更新而非重复创建
3. **回溯能力**：可通过 `doc_id` 从 MongoDB 获取完整原文
4. **Neo4j 精简**：Neo4j 不再存储文本和向量，仅存储结构索引
