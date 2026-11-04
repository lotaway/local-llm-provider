"""LocalRAG facade: composition root for retrieval, chaining and query answering."""

import asyncio
import gc
import logging
import os
from typing import Callable, Dict, List, cast

import torch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_milvus import Milvus
from constants import DATA_PATH, DB_COLLECTION, DB_HOST, DB_PORT
from model_providers import LocalLLModel
from rag.document_ingestion import DocumentIngestionMixin
from rag.document_loader import identity_after_load
from rag.document_loader import load_documents as load_documents_from_path
from rag.splitting import AdaptiveTextSplitter, select_text_splitter
from rag.vectorstore_support import VectorStoreMixin
from repositories.mongodb_repository import MongoDBRepository
from repositories.neo4j_repository import Neo4jRepository
from retrievers import ESBM25Retriever, HybridRetriever, Reranker
from retrievers.chunk_filter import GraphFilteredVectorRetriever
from retrievers.graph_coordinator import GraphQueryCoordinator
from retrievers.graph_retriever import GraphRetriever
from services.feedback_service import FeedbackService
from services.graph_extraction_service import GraphExtractionService

logger = logging.getLogger(__name__)


class LocalRAG(DocumentIngestionMixin, VectorStoreMixin):
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
        self._init_settings(llm, data_path, use_hybrid_search, use_reranking, retrieval_strategy)
        self._init_storage(use_hybrid_search)
        self._init_graph_components()
        self.last_retrieved_docs = []
        self.last_retrieved_chunk_ids = []

    def _init_settings(self, llm, data_path, use_hybrid_search, use_reranking, retrieval_strategy) -> None:
        self.llm = llm
        self.data_path = data_path
        self.reranker = Reranker() if use_reranking else None
        self.host = DB_HOST
        self.port = DB_PORT
        self.collection = DB_COLLECTION
        self.connection_uri = f"http://{self.host}:{self.port}"
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.retrieval_strategy = retrieval_strategy
        self._embeddings = None

    def _init_storage(self, use_hybrid_search: bool) -> None:
        self.mongo_repo = MongoDBRepository()
        self.es_retriever = ESBM25Retriever() if use_hybrid_search else None

    def _init_graph_components(self) -> None:
        self.neo4j_repo = Neo4jRepository()
        self.graph_extractor = GraphExtractionService(self.llm)
        self.feedback_service = FeedbackService(self.mongo_repo)
        self.graph_coordinator = GraphQueryCoordinator(self.neo4j_repo)
        self.graph_retriever = GraphRetriever(
            self.neo4j_repo, self.llm, feedback_service=self.feedback_service
        )

    def load_documents(self, after_doc_load: Callable | None = None) -> List[Document]:
        return load_documents_from_path(self.data_path, after_doc_load or identity_after_load)

    def _select_text_splitter(self, source, content_type):
        return select_text_splitter(source, content_type)

    async def init_rag_chain(self):
        vectorstore = await self.get_or_create_vectorstore()
        if vectorstore is None:
            raise RuntimeError("No vectorstore available, maybe no docs are loaded")
        vectorstore = cast(Milvus, vectorstore)
        retriever = self._build_retriever(vectorstore)
        self._install_retrieval_runnable(retriever)
        self.rag_chain = RunnableSequence(self._build_prompt_chain())

    def _build_retriever(self, vectorstore):
        k = 10 if self.use_reranking else 5
        if self.use_hybrid_search and self.es_retriever:
            return HybridRetriever(
                vectorstore=vectorstore,
                bm25_retriever=self.es_retriever,
                vector_weight=0.7,
                bm25_weight=0.3,
                k=k,
                feedback_service=self.feedback_service,
                chunk_id_provider=self.graph_coordinator.find_chunk_ids,
            )
        if self.graph_coordinator is not None:
            return GraphFilteredVectorRetriever(
                vectorstore=vectorstore,
                k=k,
                chunk_id_provider=self.graph_coordinator.find_chunk_ids,
            )
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def _install_retrieval_runnable(self, retriever) -> None:
        if self.use_reranking and self.reranker:

            def retrieve_and_rerank(query: str) -> List[Document]:
                docs = retriever.invoke(query)
                docs = cast(Reranker, self.reranker).adaptive_rerank(query, docs, top_k=5)
                self._capture_retrieval(docs)
                return docs

            self.retrieval_runnable = RunnableLambda(retrieve_and_rerank)
            return

        def retrieve_and_capture(query: str) -> List[Document]:
            docs = retriever.invoke(query)
            self._capture_retrieval(docs)
            return docs

        self.retrieval_runnable = RunnableLambda(retrieve_and_capture)

    def _build_prompt_chain(self):
        chain = (
            {"context": self.retrieval_runnable, "question": lambda x: x}
            | ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)
            | RunnableLambda(self.llm.format_messages)
            | self._chat_runnable()
            | StrOutputParser()
            | RunnableLambda(self.llm.extract_after_think)
        )
        return chain

    def _chat_runnable(self) -> RunnableLambda:
        async def chat_async(x):
            messages = x.messages if hasattr(x, "messages") else x
            return await self.llm.chat_at_once(messages, temperature=0.1, top_p=0.95, top_k=40)

        return RunnableLambda(chat_async)

    def _capture_retrieval(self, docs: List[Document]) -> None:
        self.last_retrieved_docs = docs or []
        self.last_retrieved_chunk_ids = [
            d.metadata.get("chunk_id")
            for d in self.last_retrieved_docs
            if d.metadata and d.metadata.get("chunk_id")
        ]

    async def generate_answer(self, question, stream_callback=None) -> Dict[str, str]:
        if self.rag_chain is None:
            await self.init_rag_chain()
        if stream_callback is None:
            full_response = await cast(RunnableSequence, self.rag_chain).ainvoke(question)
            return {
                "answer": self.llm.extract_after_think(full_response),
                "thought": self.llm.extract_thought(full_response),
            }
        return await self._generate_stream(question, stream_callback)

    async def _generate_stream(self, question, stream_callback) -> Dict[str, str]:
        retrieval_runnable = getattr(self, "retrieval_runnable", None)
        if not retrieval_runnable:
            await self.init_rag_chain()
            retrieval_runnable = self.retrieval_runnable

        context = await self._build_context(retrieval_runnable, question)
        prompt_value = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE).invoke(
            {"context": context, "question": question}
        )
        messages = self.llm.format_messages(prompt_value)
        full_response = await self._stream_llm(messages, stream_callback)

        return {
            "answer": self.llm.extract_after_think(full_response),
            "thought": self.llm.extract_thought(full_response),
        }

    async def _stream_llm(self, messages, stream_callback) -> str:
        full_response = ""
        async for chunk in self.llm.chat(messages, temperature=0.1):
            if isinstance(chunk, str) and chunk:
                full_response += chunk
                await self._notify_rag_stream(stream_callback, chunk)
        return full_response

    async def _build_context(self, retrieval_runnable, question) -> str:
        context_docs = await asyncio.to_thread(retrieval_runnable.invoke, question)
        context = "\n\n".join([doc.page_content for doc in context_docs])
        graph_context = await self.graph_retriever.retrieve_context(question)
        if graph_context:
            context = f"{context}\n\n{graph_context}"
        return context

    async def _notify_rag_stream(self, callback, chunk):
        if not callback:
            return
        if asyncio.iscoroutinefunction(callback):
            await callback(chunk)
        else:
            callback(chunk)

    def get_document_context(self, doc_id: str) -> str:
        doc = self.mongo_repo.get_document(doc_id)
        if doc:
            return doc.get("content", "")
        return ""

    def get_chunk_context(self, chunk_id: str) -> str:
        chunk = self.mongo_repo.get_chunk(chunk_id)
        if chunk:
            return chunk.get("text", "")
        return ""

    def get_surrounding_chunks(self, chunk_id: str, before: int = 1, after: int = 1) -> List[str]:
        chunk = self.mongo_repo.get_chunk(chunk_id)
        if not chunk:
            return []
        all_chunks = self.mongo_repo.get_chunks_by_doc(chunk["doc_id"])
        start_idx = max(0, chunk["index"] - before)
        end_idx = min(len(all_chunks), chunk["index"] + after + 1)
        return [c["text"] for c in all_chunks[start_idx:end_idx]]

    def release_memory(self):
        if hasattr(self, "rag_chain"):
            del self.rag_chain
            self.rag_chain = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("RAG GPU memory released")

    def close(self):
        if hasattr(self, "mongo_repo"):
            self.mongo_repo.close()
        if hasattr(self, "neo4j_repo"):
            self.neo4j_repo.close()
        self.release_memory()
        logger.info("All database connections closed")


def command_line_rag():
    async def run():
        local_model = LocalLLModel()
        local_rag = LocalRAG(local_model)

        logger.info("RAG system started, input 'exit' or 'quit' to quit")
        while True:
            try:
                query = await asyncio.to_thread(input, "\n问：")
                if query.lower() in ["exit", "quit"]:
                    break
                res = await local_rag.generate_answer(query)
                logger.info("RAG answer: %s", res.get("answer"))
            except KeyboardInterrupt:
                break
            except Exception as error:
                logger.warning("RAG command failed", extra={"error": str(error)})

    asyncio.run(run())


if __name__ == "__main__":
    command_line_rag()
