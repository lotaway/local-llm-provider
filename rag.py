import os
import json
import re
import logging
import hashlib
from pymilvus import connections, utility
from constants import DATA_PATH, DB_HOST, DB_PORT, DB_COLLECTION

logger = logging.getLogger(__name__)
import gc
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from model_providers import LocalLLModel
from typing import Callable, List, cast
from retrievers import HybridRetriever, Reranker, ESBM25Retriever
from file_loaders import ChatGPTLoader, DeepSeekLoader
from repositories.neo4j_repository import Neo4jRepository
from repositories.mongodb_repository import MongoDBRepository
from services.graph_extraction_service import GraphExtractionService
from services.feedback_service import FeedbackService
from retrievers.graph_retriever import GraphRetriever
import asyncio


class AdaptiveTextSplitter(TextSplitter):
    def __init__(self, min_chunk=500, max_chunk=2000, chunk_overlap=0, **kwargs):
        super().__init__(**kwargs)
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = re.split(r"\n\s*\n", text)
        chunks = [c.strip() for c in chunks if c.strip()]

        if not chunks:
            return [text] if text else []

        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk:
                sentences = re.split(r"(?<=[.!?。！？])\s+", chunk)
                current_temp = ""
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    if len(current_temp) + len(sentence) <= self.max_chunk:
                        current_temp += sentence + " "
                    else:
                        if current_temp:
                            final_chunks.append(current_temp.strip())

                        overlap = ""
                        if self.chunk_overlap > 0 and final_chunks:
                            overlap = final_chunks[-1][-self.chunk_overlap :]
                        current_temp = overlap + sentence + " "
                if current_temp:
                    final_chunks.append(current_temp.strip())

            elif len(chunk) < self.min_chunk:
                if (
                    final_chunks
                    and len(final_chunks[-1]) + len(chunk) + 2 <= self.max_chunk
                ):
                    final_chunks[-1] += "\n\n" + chunk
                else:
                    overlap = ""
                    if self.chunk_overlap > 0 and final_chunks:
                        overlap = final_chunks[-1][-self.chunk_overlap :]
                    final_chunks.append(overlap + chunk)
            else:
                overlap = ""
                if self.chunk_overlap > 0 and final_chunks:
                    overlap = final_chunks[-1][-self.chunk_overlap :]
                final_chunks.append(overlap + chunk)

        return final_chunks


class LocalRAG:
    RAG_PROMPT_TEMPLATE = """
你是检索增强问答助手，严格基于提供的内容回答问题。

【核心规则】
1. 仅根据 context 回答，禁止推测或使用外部知识，禁止提供额外建议
2. 内容不足时回复：`无法从提供的内容中找到答案`
3. 保持简洁准确，结构清晰
4. context 有矛盾时，选择最明确且重复的部分
5. 保留列表、步骤等原有格式
6. 不输出 context 原文（除非问题要求引用）
7. 问题与 context 无关时回复：`该问题与提供的内容无关`

【特殊处理：关键词查询】
如果 question 是简短的关键词（1-3个词，无问号），则：
- 检查 context 中是否包含该关键词
- 如果包含，提取并总结所有相关信息，按主题分类列出
- 格式：对找到的信息整理合并轻重有序地进行说明，关键信息分点列出"

【context】
{context}

【question】
{question}

【输出要求】
- 完整问题：直接回答，保持礼貌用词，无需开场白
- 关键词查询：按上述特殊处理方式输出
- 无相关内容：使用规则2或7的标准回复
            """

    rag_chain: RunnableSequence | None = None

    def __init__(
        self,
        llm: LocalLLModel,
        data_path=DATA_PATH,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        retrieval_strategy: str = "adaptive",
    ):
        self.llm = llm
        self.data_path = data_path
        self.reranker = Reranker() if use_reranking else None

        # Milvus connection details
        self.host = DB_HOST
        self.port = DB_PORT
        self.collection = DB_COLLECTION
        self.connection_uri = f"http://{self.host}:{self.port}"

        # Enhanced retrieval settings
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.retrieval_strategy = retrieval_strategy

        # Initialize MongoDB as source of truth
        self.mongo_repo = MongoDBRepository()

        # Initialize ES Retriever
        self.es_retriever = ESBM25Retriever() if use_hybrid_search else None

        # Graph RAG initialization
        self.neo4j_repo = Neo4jRepository()
        self.graph_extractor = GraphExtractionService(llm)
        self.graph_retriever = GraphRetriever(self.neo4j_repo, llm)

        # Feedback service for memory lifecycle
        self.feedback_service = FeedbackService(self.mongo_repo)

        # Track latest retrievals for feedback/evolution
        self.last_retrieved_docs = []
        self.last_retrieved_chunk_ids = []

    async def init_rag_chain(self):
        vectorstore = await self.get_or_create_vectorstore()
        if vectorstore is None:
            Exception("No vectorstore available, maybe not docs are loaded?")
        vectorstore = cast(Milvus, vectorstore)
        # retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        # retriever = ExpandedRetriever(vectorstore, search_kwargs={"k": 3})
        # Choose retrieval strategy
        if self.use_hybrid_search and self.es_retriever:
            retriever = HybridRetriever(
                vectorstore=vectorstore,
                bm25_retriever=self.es_retriever,
                vector_weight=0.7,
                bm25_weight=0.3,
                k=10 if self.use_reranking else 5,
                feedback_service=self.feedback_service,
            )
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 10 if self.use_reranking else 5}
            )
        # Add reranking step if enabled
        if self.use_reranking and self.reranker:

            def retrieve_and_rerank(query: str) -> List[Document]:
                docs = retriever.invoke(query)
                docs = cast(Reranker, self.reranker).adaptive_rerank(
                    query, docs, top_k=5
                )
                self._capture_retrieval(docs)
                return docs

            self.retrieval_runnable = RunnableLambda(retrieve_and_rerank)
        else:

            def retrieve_and_capture(query: str) -> List[Document]:
                docs = retriever.invoke(query)
                self._capture_retrieval(docs)
                return docs

            self.retrieval_runnable = RunnableLambda(retrieve_and_capture)

        format_messages_runnable = RunnableLambda(self.llm.format_messages)

        def prepare_messages(input_data) -> list[dict]:
            if hasattr(input_data, "messages"):
                return input_data.messages
            else:
                return input_data

        async def chat_async(x):
            return await self.llm.chat_at_once(
                prepare_messages(x),
                # do_sample=False,
                temperature=0.1,
                top_p=0.95,
                top_k=40,
            )

        chat_runnable = RunnableLambda(chat_async)
        after_runnable = RunnableLambda(self.llm.extract_after_think)

        prompt_str = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)

        chain = (
            {"context": self.retrieval_runnable, "question": lambda x: x}
            | prompt_str
            | format_messages_runnable
            | chat_runnable
            | StrOutputParser()
            | after_runnable
        )
        self.rag_chain = RunnableSequence(chain)

    def _capture_retrieval(self, docs: List[Document]) -> None:
        self.last_retrieved_docs = docs or []
        self.last_retrieved_chunk_ids = [
            d.metadata.get("chunk_id")
            for d in self.last_retrieved_docs
            if d.metadata and d.metadata.get("chunk_id")
        ]

    async def add_document(
        self, title: str, content: str, source: str, content_type: str = "md", **kwargs
    ):
        filename = source if source else f"{title}.{content_type}"
        filename = os.path.basename(filename)  # Basic protection
        file_path = os.path.join(self.data_path, "uploads", filename)

        # Ensure directory exists
        os.makedirs(os.path.join(self.data_path, "uploads"), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Calculate checksum
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Generate doc_id
        doc_id = f"doc_{checksum[:16]}"

        metadata = kwargs.get("metadata", {})
        doc_title = title or "Untitled"
        doc_author = kwargs.get("author", "")
        doc_summary = kwargs.get("summary", "")
        metadata.update({k: v for k, v in kwargs.items() if v is not None})

        self.mongo_repo.save_document(
            doc_id=doc_id,
            content=content,
            title=doc_title,
            source=source,
            author=doc_author,
            summary=doc_summary,
            path=file_path,
            filename=filename,
            format=content_type,
            checksum=f"sha256:{checksum}",
            metadata=metadata,
        )

        logger.info(f"Document {doc_id} saved to MongoDB")

        # Step 2: Generate chunks and save to MongoDB
        doc = Document(
            page_content=content,
            metadata={"source": source, "title": title, "doc_id": doc_id},
        )
        text_splitter = self._select_text_splitter(source, content_type)
        langchain_chunks = text_splitter.split_documents([doc])

        # Save chunks to MongoDB with stable IDs
        chunk_records = []
        for idx, chunk in enumerate(langchain_chunks):
            chunk_id = f"{doc_id}_chunk_{idx:04d}"

            chunk_record = self.mongo_repo.save_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                index=idx,
                text=chunk.page_content,
                offset_start=0,  # Could calculate actual offsets if needed
                offset_end=len(chunk.page_content),
                metadata=chunk.metadata,
            )
            chunk_records.append(chunk_record)

            # Update langchain chunk metadata with chunk_id
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["doc_id"] = doc_id

        self.mongo_repo.update_document_status(doc_id, "chunked", True)
        logger.info(f"Saved {len(chunk_records)} chunks to MongoDB")

        # Step 3: Vectorize and save to Milvus
        vectorstore = await self.get_or_create_vectorstore()
        if vectorstore:
            # Add documents with chunk_id in metadata
            vectorstore.add_documents(langchain_chunks)
            logger.info(f"Added {len(langchain_chunks)} chunks to Milvus")

            # Update embedding status in MongoDB
            for chunk_record in chunk_records:
                self.mongo_repo.update_chunk_embedding_status(
                    chunk_record["chunk_id"], "done"
                )

        self.mongo_repo.update_document_status(doc_id, "embedded", True)

        # Step 4: Index to Elasticsearch
        if self.use_hybrid_search and self.es_retriever:
            self.es_retriever.index_documents(langchain_chunks)
            logger.info(f"Added {len(langchain_chunks)} chunks to Elasticsearch")

        # Step 5: Extract graph and save to Neo4j (with chunk_id references)
        await self._async_extract_graph(langchain_chunks, doc_id)
        self.mongo_repo.update_document_status(doc_id, "graphed", True)

        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunk_records)}

    async def _async_extract_graph(self, chunks: List[Document], source_doc_id: str):
        logger.info(f"Starting graph extraction for {len(chunks)} chunks...")

        # Create Document node in Neo4j
        doc_metadata = chunks[0].metadata if chunks else {}
        doc_title = doc_metadata.get("title", source_doc_id)

        self.neo4j_repo.session.run(
            """
            MERGE (d:Document {doc_id: $doc_id})
            SET d.filename = $filename
            """,
            doc_id=source_doc_id,
            filename=doc_title,
        )

        total_entities = 0
        total_relations = 0

        for i, chunk in enumerate(chunks):
            try:
                chunk_id = chunk.metadata.get(
                    "chunk_id", f"{source_doc_id}_chunk_{i:04d}"
                )
                logger.info(
                    f"Processing chunk {i + 1}/{len(chunks)} (chunk_id: {chunk_id})..."
                )

                # Create Chunk pointer node in Neo4j (ID only, no text)
                self.neo4j_repo.session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.doc_id = $doc_id, c.index = $index
                    """,
                    chunk_id=chunk_id,
                    doc_id=source_doc_id,
                    index=i,
                )

                # Create relationship: Document -> Chunk
                self.neo4j_repo.session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (d)-[:CONTAINS]->(c)
                    """,
                    doc_id=source_doc_id,
                    chunk_id=chunk_id,
                )

                # Extract entities and relations from chunk text
                entities, relations = await self.graph_extractor.extract_graph(
                    chunk.page_content
                )
                logger.info(
                    f"Extracted {len(entities)} entities and {len(relations)} relations from chunk {i + 1}"
                )

                for entity in entities:
                    self.neo4j_repo.merge_entity(entity)
                    total_entities += 1

                    # Create relationship: Chunk -> Concept
                    self.neo4j_repo.session.run(
                        """
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MATCH (e:Concept {stable_id: $stable_id})
                        MERGE (c)-[:DESCRIBES]->(e)
                        """,
                        chunk_id=chunk_id,
                        stable_id=entity.stable_id,
                    )

                # Merge concept relations
                for relation in relations:
                    relation.source_doc_id = source_doc_id
                    self.neo4j_repo.merge_relation(relation)
                    total_relations += 1

            except Exception as e:
                logger.error(f"Error during graph extraction for chunk {i + 1}: {e}")
                import traceback

                traceback.print_exc()

        logger.info(
            f"Graph extraction completed. Total entities: {total_entities}, Total relations: {total_relations}"
        )

    def check_document_exists(self, bvid: str, cid: int) -> bool:
        if not self.es_retriever:
            return False

        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"metadata.bvid": bvid}},
                            {"match": {"metadata.cid": cid}},
                        ]
                    }
                },
                "size": 1,
            }
            res = self.es_retriever.es_client.search(
                index=self.es_retriever.index_name, body=query
            )
            return len(res["hits"]["hits"]) > 0
        except Exception as e:
            logger.error(f"Error checking document existence in ES: {e}")
            return False

    def load_documents(
        self,
        after_doc_load: Callable[[List[Document], str], List[Document]] = lambda x,
        _: x,
    ) -> list[Document]:
        """加载多种格式的文档"""
        docs = []

        logger.info(f"开始扫描文档目录: {self.data_path}")

        for root, _, files in os.walk(self.data_path):
            logger.info(f"当前目录: {root}, 文件数: {len(files)}")
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                logger.info(
                    f"处理文件: {file}, 扩展名: {file_ext}, 完整路径: {file_path}"
                )

                try:
                    if file_ext in [".txt"]:
                        logger.info(f"加载文本文件: {file}")
                        loader = TextLoader(file_path, encoding="utf-8")
                        loaded = loader.load()
                        logger.info(f"成功加载 {len(loaded)} 个文本片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".md", ".markdown"]:
                        logger.info(f"加载Markdown文件: {file}")
                        try:
                            loader = UnstructuredMarkdownLoader(file_path)
                            loaded = loader.load()
                        except Exception as e:
                            logger.warning(
                                f"UnstructuredMarkdownLoader 加载失败，回退到 TextLoader: {e}"
                            )
                            loader = TextLoader(file_path, encoding="utf-8")
                            loaded = loader.load()
                        logger.info(f"成功加载 {len(loaded)} 个Markdown片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".pdf":
                        logger.info(f"加载PDF文件: {file}")
                        loader = PyPDFLoader(file_path)
                        loaded = loader.load()
                        logger.info(f"成功加载 {len(loaded)} 个PDF页面")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".docx", ".doc"]:
                        logger.info(f"加载Word文档: {file}")
                        loader = UnstructuredWordDocumentLoader(file_path)
                        loaded = loader.load()
                        logger.info(f"成功加载 {len(loaded)} 个Word文档片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".pptx", ".ppt"]:
                        logger.info(f"加载PPT文件: {file}")
                        loader = UnstructuredPowerPointLoader(file_path)
                        loaded = loader.load()
                        logger.info(f"成功加载 {len(loaded)} 个PPT幻灯片")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".xlsx", ".xls"]:
                        logger.info(f"加载Excel文件: {file}")
                        loader = UnstructuredExcelLoader(file_path)
                        loaded = loader.load()
                        logger.info(f"成功加载 {len(loaded)} 个Excel工作表")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".csv":
                        logger.info(f"加载CSV文件: {file}")
                        loader = CSVLoader(file_path)
                        loaded = loader.load()
                        logger.info(f"成功加载 {len(loaded)} 个CSV记录")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".json":
                        logger.info(f"加载JSON文件: {file}")
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            gpt_loader = ChatGPTLoader()
                            ds_loader = DeepSeekLoader()
                            if gpt_loader.check(data):
                                logger.info(f"检测到 ChatGPT 导出格式: {file}")
                                loaded = gpt_loader.load(data, file)
                                logger.info(
                                    f"成功解析 ChatGPT 对话，生成 {len(loaded)} 个文档片段"
                                )
                                docs.extend(after_doc_load(loaded, file))

                            elif ds_loader.check(data):
                                logger.info(f"检测到 DeepSeek 导出格式: {file}")
                                loaded = ds_loader.load(data, file)
                                logger.info(
                                    f"成功解析 DeepSeek 对话，生成 {len(loaded)} 个文档片段"
                                )
                                docs.extend(after_doc_load(loaded, file))

                            else:
                                # Default JSON loading
                                loader = JSONLoader(
                                    file_path=file_path,
                                    jq_schema=".",
                                    text_content=True,
                                )
                                loaded = loader.load()
                                logger.info(f"成功加载 {len(loaded)} 个JSON条目")
                                docs.extend(after_doc_load(loaded, file))

                        except Exception as e:
                            logger.error(f"解析 JSON 文件 {file} 失败: {e}")
                            # Fallback
                            try:
                                loader = JSONLoader(
                                    file_path=file_path,
                                    jq_schema=".",
                                    text_content=True,
                                )
                                loaded = loader.load()
                                docs.extend(after_doc_load(loaded, file))
                            except Exception as e2:
                                logger.error(f"回退加载也失败: {e2}")

                    elif file_ext in [
                        ".py",
                        ".java",
                        ".kt",
                        ".rs",
                        ".js",
                        ".ts",
                        ".html",
                        ".css",
                        ".cs",
                        ".swift",
                    ]:
                        logger.info(f"加载代码文件: {file}")
                        loader = TextLoader(file_path, encoding="utf-8")
                        loaded = loader.load()
                        logger.info(f"成功加载 {len(loaded)} 个代码片段")
                        docs.extend(after_doc_load(loaded, file))

                    else:
                        logger.warning(f"跳过不支持的文件格式: {file}")

                except Exception as e:
                    logger.error(f"加载文件 {file} 时出错: {str(e)}")
                    import traceback

                    traceback.print_exc()

        logger.info(f"成功加载 {len(docs)} 个文档片段")
        return docs

    def get_embeddings(self):
        logger.info("Initializing embedding model")
        if hasattr(self, "_embeddings") and self._embeddings is not None:
            return self._embeddings

        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        logger.info(f"Loading embedding model on {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 8},
        )
        self._embeddings = embeddings
        return embeddings

    def build_vectorstore(self, docs: list[Document]):
        all_chunks = []

        # Process each document
        for doc in docs:
            # Generate doc_id from content checksum
            content = doc.page_content
            checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
            doc_id = f"doc_{checksum[:16]}"

            # Save document to MongoDB if not exists
            source = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source) if source else "unknown"
            doc_title = doc.metadata.get("title") or filename or "Untitled"
            doc_author = doc.metadata.get("author", "")
            doc_summary = doc.metadata.get("summary", "")
            if not self.mongo_repo.get_document(doc_id):
                self.mongo_repo.save_document(
                    doc_id=doc_id,
                    content=content,
                    title=doc_title,
                    source=source,
                    author=doc_author,
                    summary=doc_summary,
                    path=source,
                    filename=filename,
                    format="auto",
                    checksum=f"sha256:{checksum}",
                    metadata=doc.metadata,
                )

            # Split into chunks
            splitter = self._select_text_splitter(
                doc.metadata.get("source"), doc.metadata.get("content_type")
            )
            doc_chunks = splitter.split_documents([doc])

            # Save chunks to MongoDB with stable IDs
            for idx, chunk in enumerate(doc_chunks):
                chunk_id = f"{doc_id}_chunk_{idx:04d}"

                # Add chunk_id to metadata
                chunk.metadata["chunk_id"] = chunk_id
                chunk.metadata["doc_id"] = doc_id

                # Save to MongoDB
                self.mongo_repo.save_chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    index=idx,
                    text=chunk.page_content,
                    offset_start=0,
                    offset_end=len(chunk.page_content),
                    metadata=chunk.metadata,
                )

                all_chunks.append(chunk)

            # Update document status
            self.mongo_repo.update_document_status(doc_id, "chunked", True)

        logger.info(f"共切分为 {len(all_chunks)} 个文本块，开始分批向量化...")

        # 分批处理以避免显存溢出
        batch_size = 50
        vectorstore: Milvus | None = None

        embeddings = self.get_embeddings()

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            logger.info(
                f"正在处理第 {i + 1} 到 {min(i + batch_size, len(all_chunks))} 个文本块..."
            )

            if vectorstore is None:
                vectorstore = Milvus.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    connection_args={"uri": self.connection_uri},
                    collection_name=self.collection,
                )
            else:
                vectorstore.add_documents(batch)

            # Update embedding status in MongoDB
            for chunk in batch:
                chunk_id = chunk.metadata.get("chunk_id")
                if chunk_id:
                    self.mongo_repo.update_chunk_embedding_status(chunk_id, "done")

            logger.info(
                f"完成第 {i + 1} 到 {min(i + batch_size, len(all_chunks))} 个文本块的向量化"
            )

            # 清理显存
            del batch
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("所有文本块向量化完成")

        return vectorstore

    def _sync_get_or_create_vectorstore(self, main_loop: asyncio.AbstractEventLoop):
        """
        Synchronous version of get_or_create_vectorstore.
        This should be called from a thread pool.
        """
        connections.connect(uri=self.connection_uri)
        embeddings = self.get_embeddings()

        has_milvus = utility.has_collection(self.collection)
        has_mongo_docs = self.mongo_repo.documents.count_documents({}) > 0
        has_graph = not self.neo4j_repo.is_empty()
        has_es_docs = False
        if self.es_retriever and self.es_retriever.es_client:
            try:
                has_es_docs = self.es_retriever.es_client.indices.exists(
                    index=self.es_retriever.index_name
                )
            except Exception:
                pass

        # Calculate which stores need initialization
        needs_milvus = has_milvus and has_mongo_docs
        needs_es = (
            self.use_hybrid_search
            and self.es_retriever
            and has_mongo_docs
            and not has_es_docs
        )
        needs_graph = has_mongo_docs and not has_graph

        if has_milvus and has_graph and has_es_docs:
            logger.info(f"All stores already exist, reusing them.")
            return Milvus(
                embedding_function=embeddings,
                connection_args={"uri": self.connection_uri},
                collection_name=self.collection,
            )

        logger.info(
            f"Initialization status: Milvus={has_milvus}, MongoDB={has_mongo_docs}, "
            f"ES={has_es_docs}, Graph={has_graph}"
        )

        docs = self.load_documents()
        if not docs:
            if not has_milvus:
                raise ValueError(
                    f"在路径 {self.data_path} 中没有找到任何文档，无法初始化向量库"
                )
            else:
                logger.warning(
                    f"Warning: No documents found in {self.data_path}, but Milvus collection exists. Returning existing collection."
                )
                return Milvus(
                    embedding_function=embeddings,
                    connection_args={"uri": self.connection_uri},
                    collection_name=self.collection,
                )

        chunks = self._split_documents(docs)

        vectorstore = None

        # Step 1: Ensure MongoDB has all documents and chunks
        logger.info("Ensuring MongoDB is up to date...")
        self._sync_ensure_mongodb(docs)

        # Step 2: Handle Milvus
        if not has_milvus:
            logger.info(
                f"Creating Milvus collection '{self.collection}' and indexing documents..."
            )
            if needs_es and self.es_retriever:
                logger.info("Indexing documents to Elasticsearch...")
                self.es_retriever.index_documents(chunks)
            vectorstore = self.build_vectorstore(docs)
        else:
            logger.info(
                f"Milvus collection '{self.collection}' exists, checking for missing chunks..."
            )
            # Check for and add missing chunks only
            vectorstore = Milvus(
                embedding_function=embeddings,
                connection_args={"uri": self.connection_uri},
                collection_name=self.collection,
            )
            # Sync any missing chunks to Milvus
            vectorstore = self._sync_add_missing_chunks(vectorstore, chunks)

        # Step 3: Handle Elasticsearch
        if needs_es and self.es_retriever:
            logger.info("Indexing documents to Elasticsearch...")
            self.es_retriever.index_documents(chunks)

        # Step 4: Handle Graph extraction
        if needs_graph:
            logger.info("Scheduling graph extraction in background...")
            asyncio.run_coroutine_threadsafe(
                self._async_extract_graph(chunks, "initial_load"), main_loop
            )

        return vectorstore

    def _sync_ensure_mongodb(self, docs: list[Document]):
        """Ensure MongoDB has all documents and chunks, skip if already exists"""
        for doc in docs:
            content = doc.page_content
            checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
            doc_id = f"doc_{checksum[:16]}"

            # Check if document already exists
            if self.mongo_repo.get_document(doc_id):
                logger.debug(f"Document {doc_id} already exists in MongoDB, skipping")
                continue

            # Save new document
            source = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source) if source else "unknown"
            doc_title = doc.metadata.get("title") or filename or "Untitled"
            doc_author = doc.metadata.get("author", "")
            doc_summary = doc.metadata.get("summary", "")
            self.mongo_repo.save_document(
                doc_id=doc_id,
                content=content,
                title=doc_title,
                source=source,
                author=doc_author,
                summary=doc_summary,
                path=source,
                filename=filename,
                format="auto",
                checksum=f"sha256:{checksum}",
                metadata=doc.metadata,
            )

            # Split and save chunks
            splitter = self._select_text_splitter(
                doc.metadata.get("source"), doc.metadata.get("content_type")
            )
            doc_chunks = splitter.split_documents([doc])
            for idx, chunk in enumerate(doc_chunks):
                chunk_id = f"{doc_id}_chunk_{idx:04d}"

                # Check if chunk exists
                if self.mongo_repo.get_chunk(chunk_id):
                    continue

                chunk.metadata["chunk_id"] = chunk_id
                chunk.metadata["doc_id"] = doc_id

                self.mongo_repo.save_chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    index=idx,
                    text=chunk.page_content,
                    offset_start=0,
                    offset_end=len(chunk.page_content),
                    metadata=chunk.metadata,
                )

            self.mongo_repo.update_document_status(doc_id, "chunked", True)
            logger.info(f"Added new document {doc_id} to MongoDB")

    def _sync_add_missing_chunks(
        self, vectorstore: Milvus, all_chunks: list[Document]
    ) -> Milvus:
        """Add only chunks that don't exist in Milvus yet"""
        embeddings = self.get_embeddings()
        missing_chunks = []

        for chunk in all_chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            if chunk_id:
                # Check MongoDB for embedding status
                mongo_chunk = self.mongo_repo.get_chunk(chunk_id)
                if mongo_chunk and mongo_chunk.get("embedding_status") != "done":
                    missing_chunks.append(chunk)

        if missing_chunks:
            logger.info(f"Adding {len(missing_chunks)} missing chunks to Milvus...")

            batch_size = 50
            for i in range(0, len(missing_chunks), batch_size):
                batch = missing_chunks[i : i + batch_size]
                vectorstore.add_documents(batch)

                # Update status
                for c in batch:
                    cid = c.metadata.get("chunk_id")
                    if cid:
                        self.mongo_repo.update_chunk_embedding_status(cid, "done")
        else:
            logger.info("No missing chunks to add to Milvus")

        return vectorstore

    def _select_text_splitter(
        self, source: str | None, content_type: str | None
    ) -> TextSplitter:
        file_ext = ""
        if content_type:
            file_ext = f".{content_type.lstrip('.')}".lower()
        elif source:
            file_ext = os.path.splitext(source)[1].lower()

        if file_ext in [".md", ".markdown"]:
            return RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                separators=[
                    "\n## ",
                    "\n# ",
                    "\n\n",
                    "\n",
                    "。",
                    "！",
                    "？",
                    ".",
                    "!",
                    "?",
                    ";",
                    "；",
                    ",",
                    "，",
                    " ",
                ],
            )

        if file_ext in [
            ".py",
            ".java",
            ".kt",
            ".rs",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".cs",
            ".swift",
        ]:
            return RecursiveCharacterTextSplitter(
                chunk_size=1800,
                chunk_overlap=200,
                separators=[
                    "\nclass ",
                    "\ndef ",
                    "\nfunction ",
                    "\n\n",
                    "\n",
                    " ",
                ],
            )

        if file_ext in [".json", ".csv", ".xlsx", ".xls"]:
            return RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=100,
                separators=["\n\n", "\n", ",", " "],
            )

        return AdaptiveTextSplitter(min_chunk=500, max_chunk=2000, chunk_overlap=200)

    def _split_documents(self, docs: list[Document]) -> list[Document]:
        all_chunks = []
        for doc in docs:
            splitter = self._select_text_splitter(
                doc.metadata.get("source"), doc.metadata.get("content_type")
            )
            all_chunks.extend(splitter.split_documents([doc]))
        return all_chunks

    async def get_or_create_vectorstore(self):
        # Run all synchronous blocking operations in a thread pool
        # to avoid blocking the async event loop
        import asyncio

        logger.info("Loading vectorstore (this may take a while)...")
        loop = asyncio.get_running_loop()
        vectorstore = await asyncio.to_thread(
            self._sync_get_or_create_vectorstore, loop
        )
        return vectorstore

    async def generate_answer(self, question, stream_callback=None):
        if self.rag_chain is None:
            await self.init_rag_chain()

        if stream_callback is None:
            # Non-streaming mode - use the existing chain with await
            return await cast(RunnableSequence, self.rag_chain).ainvoke(question)
        else:
            logger.info("Starting RAG generation...")
            if hasattr(self, "retrieval_runnable"):
                retrieval_runnable = self.retrieval_runnable
            else:
                logger.info("Initializing RAG chain for retrieval...")
                await self.init_rag_chain()
                retrieval_runnable = self.retrieval_runnable
            logger.info("Retrieving context...")
            # retrieval_runnable might be synchronous so we wrap it or just invoke if it's CPU bound but fast
            # RunnableSequence.ainvoke handles sync steps in threadpool usually
            # But here we are invoking runnable directly.
            # If retrieval_runnable is standard LangChain runnable, invoke is sync.
            # Best to run in thread if blocking.
            import asyncio

            context_docs = await asyncio.to_thread(retrieval_runnable.invoke, question)

            context = "\n\n".join([doc.page_content for doc in context_docs])

            # Augment with Graph Context
            graph_context = await self.graph_retriever.retrieve_context(question)
            if graph_context:
                logger.info("Adding graph context to prompt...")
                context = f"{context}\n\n{graph_context}"

            logger.info("Formatting prompt...")

            prompt_str = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)
            # prompt_str.invoke is fast
            prompt_value = prompt_str.invoke({"context": context, "question": question})
            messages = self.llm.format_messages(prompt_value)
            logger.info("Starting LLM chat stream...")

            full_response = ""
            logger.info("Entering stream loop...")
            try:
                async for chunk in self.llm.chat(
                    messages, temperature=0.1, top_p=0.95, top_k=40
                ):
                    if isinstance(chunk, int):
                        continue
                    if chunk:
                        if isinstance(chunk, dict):
                            text = chunk.get(
                                "content", chunk.get("reasoning_content", "")
                            )
                        else:
                            text = chunk
                        full_response += text
                        if stream_callback:
                            if asyncio.iscoroutinefunction(stream_callback):
                                await stream_callback(chunk)
                            else:
                                stream_callback(chunk)
            except Exception as e:
                logger.error(f"Stream error: {e}")
                raise e
            logger.info("Stream finished.")

            return self.llm.extract_after_think(full_response)

    def get_document_context(self, doc_id: str) -> str:
        """Retrieve full document context from MongoDB by doc_id"""
        doc = self.mongo_repo.get_document(doc_id)
        if doc:
            return doc.get("content", "")
        return ""

    def get_chunk_context(self, chunk_id: str) -> str:
        """Retrieve chunk context from MongoDB by chunk_id"""
        chunk = self.mongo_repo.get_chunk(chunk_id)
        if chunk:
            return chunk.get("text", "")
        return ""

    def get_surrounding_chunks(
        self, chunk_id: str, before: int = 1, after: int = 1
    ) -> List[str]:
        """
        Retrieve surrounding chunks for better context.
        Returns [before chunks, current chunk, after chunks]
        """
        chunk = self.mongo_repo.get_chunk(chunk_id)
        if not chunk:
            return []

        doc_id = chunk["doc_id"]
        current_index = chunk["index"]

        all_chunks = self.mongo_repo.get_chunks_by_doc(doc_id)

        start_idx = max(0, current_index - before)
        end_idx = min(len(all_chunks), current_index + after + 1)

        return [c["text"] for c in all_chunks[start_idx:end_idx]]

    def release_memory(self):
        """释放显存资源"""
        if hasattr(self, "rag_chain"):
            del self.rag_chain
            self.rag_chain = None  # Set to None after deletion
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("已释放 RAG 相关显存资源")

    def close(self):
        """Clean up all database connections"""
        if hasattr(self, "mongo_repo"):
            self.mongo_repo.close()
        if hasattr(self, "neo4j_repo"):
            self.neo4j_repo.close()
        self.release_memory()
        logger.info("所有数据库连接已关闭")


def command_line_rag():
    import asyncio

    async def run():
        local_model = LocalLLModel()
        local_rag = LocalRAG(local_model)

        print("RAG 系统已启动，输入 'exit' 或 'quit' 退出")
        while True:
            try:
                # Use thread for blocking input
                query = await asyncio.to_thread(input, "\n问：")
                if query.lower() in ["exit", "quit"]:
                    break
                answer = await local_rag.generate_answer(query)
                print("答：", answer)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")

    asyncio.run(run())


if __name__ == "__main__":
    command_line_rag()
