import os
import json
import re
from pymilvus import connections, utility
import gc
import torch
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
from langchain_text_splitters import TextSplitter

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

    def __init__(
        self,
        llm: LocalLLModel,
        data_path=os.getenv("DATA_PATH", "./docs"),
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        retrieval_strategy: str = "adaptive",
    ):
        self.llm = llm
        self.data_path = data_path
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "19530")
        self.collection = os.getenv("DB_COLLECTION", "rag_docs")
        self.rag_chain: RunnableSequence | None = None

        # Enhanced retrieval settings
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.retrieval_strategy = retrieval_strategy  # "adaptive", "hybrid", "vector"
        self.reranker = Reranker() if use_reranking else None

        # Initialize ES Retriever
        self.es_retriever = ESBM25Retriever() if use_hybrid_search else None

    def init_rag_chain(self):
        vectorstore = self.get_or_create_vectorstore()
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
            )
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 10 if self.use_reranking else 5}
            )
        # Add reranking step if enabled
        if self.use_reranking and self.reranker:

            def retrieve_and_rerank(query: str) -> List[Document]:
                docs = retriever.invoke(query)
                return cast(Reranker, self.reranker).adaptive_rerank(
                    query, docs, top_k=5
                )

            self.retrieval_runnable = RunnableLambda(retrieve_and_rerank)
        else:
            self.retrieval_runnable = retriever

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

    def add_document(
        self, title: str, content: str, source: str, content_type: str = "md", **kwargs
    ):
        """Import a single document"""
        filename = source if source else f"{title}.{content_type}"
        filename = os.path.basename(filename)  # Basic protection
        file_path = os.path.join(self.data_path, "uploads", filename)

        # Ensure directory exists
        os.makedirs(os.path.join(self.data_path, "uploads"), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Document saved to {file_path}")

        metadata = {"source": source, "title": title}
        metadata.update({k: v for k, v in kwargs.items() if v is not None})

        doc = Document(page_content=content, metadata=metadata)

        text_splitter = AdaptiveTextSplitter(
            min_chunk=500, max_chunk=2000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents([doc])
        vectorstore = self.get_or_create_vectorstore()
        if vectorstore:
            vectorstore.add_documents(chunks)
            print(f"Added {len(chunks)} chunks to Milvus")

        # Update External Search Store (Elasticsearch)
        if self.use_hybrid_search and self.es_retriever:
            self.es_retriever.index_documents(chunks)
            print(f"Added {len(chunks)} chunks to Elasticsearch")

        return {"filename": filename, "chunks": len(chunks)}

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
            print(f"Error checking document existence in ES: {e}")
            return False

    def load_base_documents(self):
        docs = []
        for root, _, files in os.walk(self.data_path):
            for f in files:
                if f.endswith(".txt") or f.endswith(".md"):
                    try:
                        loader = TextLoader(os.path.join(root, f), encoding="utf-8")
                        docs.extend(loader.load())
                    except Exception as e:
                        print(f"加载文件 {f} 时出错: {e}")
        return docs

    def load_documents(
        self,
        after_doc_load: Callable[
            [List[Document], str], List[Document]
        ] = lambda x, _: x,
    ) -> list[Document]:
        """加载多种格式的文档"""
        docs = []

        print(f"开始扫描文档目录: {self.data_path}")

        for root, _, files in os.walk(self.data_path):
            print(f"当前目录: {root}, 文件数: {len(files)}")
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                print(f"处理文件: {file}, 扩展名: {file_ext}, 完整路径: {file_path}")

                try:
                    if file_ext in [".txt"]:
                        print(f"加载文本文件: {file}")
                        loader = TextLoader(file_path, encoding="utf-8")
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个文本片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".md", ".markdown"]:
                        print(f"加载Markdown文件: {file}")
                        loader = UnstructuredMarkdownLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个Markdown片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".pdf":
                        print(f"加载PDF文件: {file}")
                        loader = PyPDFLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个PDF页面")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".docx", ".doc"]:
                        print(f"加载Word文档: {file}")
                        loader = UnstructuredWordDocumentLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个Word文档片段")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".pptx", ".ppt"]:
                        print(f"加载PPT文件: {file}")
                        loader = UnstructuredPowerPointLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个PPT幻灯片")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext in [".xlsx", ".xls"]:
                        print(f"加载Excel文件: {file}")
                        loader = UnstructuredExcelLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个Excel工作表")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".csv":
                        print(f"加载CSV文件: {file}")
                        loader = CSVLoader(file_path)
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个CSV记录")
                        docs.extend(after_doc_load(loaded, file))

                    elif file_ext == ".json":
                        print(f"加载JSON文件: {file}")
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            gpt_loader = ChatGPTLoader()
                            ds_loader = DeepSeekLoader()
                            if gpt_loader.check(data):
                                print(f"检测到 ChatGPT 导出格式: {file}")
                                loaded = gpt_loader.load(data, file)
                                print(
                                    f"成功解析 ChatGPT 对话，生成 {len(loaded)} 个文档片段"
                                )
                                docs.extend(after_doc_load(loaded, file))

                            elif ds_loader.check(data):
                                print(f"检测到 DeepSeek 导出格式: {file}")
                                loaded = ds_loader.load(data, file)
                                print(
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
                                print(f"成功加载 {len(loaded)} 个JSON条目")
                                docs.extend(after_doc_load(loaded, file))

                        except Exception as e:
                            print(f"解析 JSON 文件 {file} 失败: {e}")
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
                                print(f"回退加载也失败: {e2}")

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
                        print(f"加载代码文件: {file}")
                        loader = TextLoader(file_path, encoding="utf-8")
                        loaded = loader.load()
                        print(f"成功加载 {len(loaded)} 个代码片段")
                        docs.extend(after_doc_load(loaded, file))

                    else:
                        print(f"跳过不支持的文件格式: {file}")

                except Exception as e:
                    print(f"加载文件 {file} 时出错: {str(e)}")
                    import traceback

                    traceback.print_exc()

        print(f"成功加载 {len(docs)} 个文档片段")
        return docs

    def get_embeddings(self):
        print("Initializing embedding model")
        embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 8},
        )
        return embeddings

    def build_vectorstore(self, docs: list[Document]):
        text_splitter = AdaptiveTextSplitter(
            min_chunk=500, max_chunk=2000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        embeddings = self.get_embeddings()
        print(f"共切分为 {len(chunks)} 个文本块，开始分批向量化...")

        # 分批处理以避免显存溢出
        batch_size = 50
        vectorstore: Milvus | None = None

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(
                f"正在处理第 {i + 1} 到 {min(i + batch_size, len(chunks))} 个文本块..."
            )

            if vectorstore is None:
                vectorstore = Milvus.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    connection_args={"host": self.host, "port": self.port},
                    collection_name=self.collection,
                )
            else:
                vectorstore.add_documents(batch)

            # 清理显存
            del batch
            gc.collect()
            torch.cuda.empty_cache()

        return vectorstore

    def get_or_create_vectorstore(self):
        connections.connect(host=self.host, port=self.port)
        embeddings = self.get_embeddings()
        if not utility.has_collection(self.collection):
            print(
                f"Collection '{self.collection}' not found, creating and inserting documents..."
            )
            docs = self.load_documents()
            if not docs:
                raise ValueError(f"在路径 {self.data_path} 中没有找到任何文档")
            text_splitter = AdaptiveTextSplitter(
                min_chunk=500, max_chunk=2000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)

            if self.use_hybrid_search and self.es_retriever:
                print("Indexing documents to Elasticsearch...")
                self.es_retriever.index_documents(chunks)

            return self.build_vectorstore(docs)
        else:
            print(f"Collection '{self.collection}' already exists, reusing it.")
            return Milvus(
                embedding_function=embeddings,
                connection_args={"host": self.host, "port": self.port},
                collection_name=self.collection,
            )

    async def generate_answer(self, question, stream_callback=None):
        if self.rag_chain is None:
            self.init_rag_chain()

        if stream_callback is None:
            # Non-streaming mode - use the existing chain with await
            return await cast(RunnableSequence, self.rag_chain).ainvoke(question)
        else:
            print("Starting RAG generation...")
            if hasattr(self, "retrieval_runnable"):
                retrieval_runnable = self.retrieval_runnable
            else:
                print("Initializing RAG chain for retrieval...")
                self.init_rag_chain()
                retrieval_runnable = self.retrieval_runnable
            print("Retrieving context...")
            # retrieval_runnable might be synchronous so we wrap it or just invoke if it's CPU bound but fast
            # RunnableSequence.ainvoke handles sync steps in threadpool usually
            # But here we are invoking runnable directly.
            # If retrieval_runnable is standard LangChain runnable, invoke is sync.
            # Best to run in thread if blocking.
            import asyncio

            context_docs = await asyncio.to_thread(retrieval_runnable.invoke, question)

            print(f"Retrieved {len(context_docs)} docs")
            context = "\n\n".join([doc.page_content for doc in context_docs])
            print("Formatting prompt...")
            prompt_str = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)
            # prompt_str.invoke is fast
            prompt_value = prompt_str.invoke({"context": context, "question": question})
            messages = self.llm.format_messages(prompt_value)
            print("Starting LLM chat stream...")

            full_response = ""
            print("Entering stream loop...")
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
                print(f"Stream error: {e}")
                raise e
            print("Stream finished.")

            return self.llm.extract_after_think(full_response)

    def release_memory(self):
        """释放显存资源"""
        if hasattr(self, "rag_chain"):
            del self.rag_chain
            self.rag_chain = None  # Set to None after deletion
        gc.collect()
        torch.cuda.empty_cache()
        print("已释放 RAG 相关显存资源")


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
