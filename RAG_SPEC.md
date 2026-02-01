# 一、目标与迁移背景

## 1.1 现状

* Milvus：仅存储向量（chunk 级），与 Neo4j、ES 无直接关联
* Neo4j：单独存储结构化实体与关系
* ES：全文索引
* 原始文档：以文件形式存在于目录中，无统一事实源

## 1.2 目标架构

* MongoDB 作为 **原始文档事实源（Source of Truth）**
* Milvus 存储 **所有 chunk 的向量与文本**
* Neo4j 存储 **结构化实体、关系，以及指向 chunk_id 的索引关系**
* ES 用于 **全文 / 关键词 / 高亮检索**

核心原则：

* 文本只在 Mongo / Milvus / ES 中存在
* Neo4j 不存文本、不存向量，只存 ID 与结构

---

# 二、MongoDB 设计（原始文档层）

## 2.1 Collection：documents

```json
{
  "_id": "doc_001",
  "source": "filesystem",
  "path": "/data/docs/llm/transformer.md",
  "filename": "transformer.md",
  "format": "markdown",
  "checksum": "sha256:...",
  "created_at": "2026-01-30T10:00:00Z",
  "updated_at": "2026-01-30T10:00:00Z",
  "content": "完整原始文档文本",
  "status": {
    "chunked": true,
    "embedded": true,
    "graphed": true
  }
}
```

说明：

* `path`：记录原始文件所在目录与文件名
* `checksum`：用于判断文件是否变更
* `status`：用于控制增量处理流程

---

## 2.2 Collection：chunks

```json
{
  "chunk_id": "doc_001_chunk_0005",
  "doc_id": "doc_001",
  "index": 5,
  "offset_start": 1200,
  "offset_end": 1600,
  "text": "该 chunk 的文本内容",
  "embedding_status": "done"
}
```

说明：

* Mongo 中保留 chunk 是为了可回溯、可重建
* Milvus / ES 可由该表重建

---

# 三、Milvus 设计（向量层）

## 3.1 Collection：document_chunks

```text
Primary Key: chunk_id

Fields:
- chunk_id (string)
- doc_id (string)
- embedding (float vector)
- text (string)
- metadata (json)
```

metadata 示例：

```json
{
  "source": "doc_001",
  "index": 5
}
```

原则：

* 所有 chunk 必须进入 Milvus
* 不因是否被抽成结构而区分

---

# 四、Neo4j 设计（结构层）

## 4.1 节点类型

### Entity

```
(:Concept {stable_id, name})
(:Document {doc_id, filename})
```

### Chunk（指针节点）

```
(:Chunk {chunk_id, doc_id, index})
```

Chunk 节点仅作为 **ID 指针**，不存文本、不存向量

---

## 4.2 关系设计

```
(Document)-[:CONTAINS]->(Chunk)
(Chunk)-[:DESCRIBES]->(Concept)
(Concept)-[:DEPENDS_ON]->(Concept)
```

---

# 五、ES 设计（全文索引层）

## 5.1 Index：document_chunks

```json
{
  "chunk_id": "doc_001_chunk_0005",
  "doc_id": "doc_001",
  "text": "chunk 文本",
  "filename": "transformer.md"
}
```

用途：

* 关键词检索
* 高亮
* 精确匹配

---

# 六、完整数据处理流程（迁移后）

## 6.1 文档导入

1. 扫描指定目录
2. 读取文件内容
3. 计算 checksum
4. 写入 MongoDB.documents

---

## 6.2 Chunk 生成

1. 从 MongoDB.documents 读取 content
2. 切分为 chunk
3. 写入 MongoDB.chunks

---

## 6.3 向量化

1. 从 MongoDB.chunks 读取 text
2. 生成 embedding
3. 写入 Milvus.document_chunks
4. 同步写入 ES.document_chunks

---

## 6.4 结构抽取

1. 基于 chunk 文本调用 LLM
2. 生成 Entity / Relation JSON
3. 写入 Neo4j
4. 在 Neo4j 中创建 (Entity)-[:DESCRIBED_BY]->(Chunk)

Chunk 引用的唯一依据：chunk_id

---

# 七、查询协同方式（示例）

## 问题："Attention 在 Transformer 中如何实现"

1. Neo4j：查 Concept = Attention
2. Neo4j：找到关联 chunk_id
3. Milvus：按 chunk_id 过滤 + 向量相似度
4. ES：补充关键词结果
5. Mongo：如需完整上下文，按 doc_id 回溯

---

# 八、新文档加入的标准流程

1. 新文件写入目录
2. 扫描并写入 MongoDB.documents
3. 生成 chunk → MongoDB.chunks
4. 向量化 → Milvus + ES
5. 抽取结构 → Neo4j
6. 更新 documents.status

整个流程 **可重入、可重跑、可增量**

---

# 九、关键设计原则总结

* MongoDB 是事实源
* Milvus 保证语义完整性
* Neo4j 只做结构索引
* ES 提供全文能力
* 所有关联都通过稳定 ID（doc_id / chunk_id）完成
