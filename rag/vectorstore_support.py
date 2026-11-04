"""Milvus vectorstore construction and incremental sync for ``LocalRAG``."""

import asyncio
import gc
import logging
import os
from typing import List, NamedTuple, Optional

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility

from constants import EMBEDDING_MODEL
from rag.splitting import document_checksum, document_id, select_text_splitter, split_documents

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 50


class StoreState(NamedTuple):
    has_milvus: bool
    has_mongo_docs: bool
    has_graph: bool
    has_pending: bool
    has_incomplete: bool
    has_es_docs: bool
    needs_es: bool
    needs_graph: bool


class VectorStoreMixin:
    """Embedding model, Milvus collection and ES/graph synchronization."""

    def get_embeddings(self):
        logger.info("Initializing embedding model")
        if getattr(self, "_embeddings", None) is not None:
            return self._embeddings
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        logger.info(f"Loading embedding model on {device}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 8},
        )
        self._embeddings = embeddings
        return embeddings

    async def get_or_create_vectorstore(self):
        logger.info("Loading vectorstore (this may take a while)...")
        loop = asyncio.get_running_loop()
        return await asyncio.to_thread(self._sync_get_or_create_vectorstore, loop)

    def build_vectorstore(self, docs: List[Document]) -> Optional[Milvus]:
        all_chunks: List[Document] = []
        for doc in docs:
            all_chunks.extend(self._persist_document_chunks(doc))
        return self._embed_chunks_in_batches(all_chunks, docs)

    def _persist_document_chunks(self, doc: Document) -> List[Document]:
        doc_id = document_id(doc.page_content)
        self._sync_save_document(doc, doc_id, document_checksum(doc.page_content))
        splitter = select_text_splitter(doc.metadata.get("source"), doc.metadata.get("content_type"))
        chunks: List[Document] = []
        for index, chunk in enumerate(splitter.split_documents([doc])):
            cid = f"{doc_id}_chunk_{index:04d}"
            chunk.metadata["chunk_id"] = cid
            chunk.metadata["doc_id"] = doc_id
            self._save_vector_chunk(cid, doc_id, index, chunk)
            chunks.append(chunk)
        self.mongo_repo.update_document_status(doc_id, "chunked", True)
        return chunks

    def _save_vector_chunk(self, cid: str, doc_id: str, index: int, chunk: Document) -> None:
        self.mongo_repo.save_chunk(
            chunk_id=cid,
            doc_id=doc_id,
            index=index,
            text=chunk.page_content,
            offset_start=0,
            offset_end=len(chunk.page_content),
            metadata=chunk.metadata,
        )

    def _embed_chunks_in_batches(self, all_chunks: List[Document], docs: List[Document]) -> Optional[Milvus]:
        logger.info(f"Split into {len(all_chunks)} chunks, embedding in batches...")
        vectorstore: Optional[Milvus] = None
        embeddings = self.get_embeddings()
        for start in range(0, len(all_chunks), EMBED_BATCH_SIZE):
            batch = all_chunks[start : start + EMBED_BATCH_SIZE]
            vectorstore = self._embed_batch(vectorstore, batch, embeddings)
            self._mark_batch_embedded(batch)
            gc.collect()
            torch.cuda.empty_cache()
        for doc in docs:
            self.mongo_repo.update_document_status(document_id(doc.page_content), "embedded", True)
        logger.info("All chunks embedded")
        return vectorstore

    def _embed_batch(self, vectorstore: Optional[Milvus], batch: List[Document], embeddings) -> Milvus:
        if vectorstore is None:
            return Milvus.from_documents(
                documents=batch,
                embedding=embeddings,
                connection_args={"uri": self.connection_uri},
                collection_name=self.collection,
            )
        vectorstore.add_documents(batch)
        return vectorstore

    def _mark_batch_embedded(self, batch: List[Document]) -> None:
        for chunk in batch:
            cid = chunk.metadata.get("chunk_id")
            if cid:
                self.mongo_repo.update_chunk_embedding_status(cid, "done")

    def _open_milvus(self, embeddings) -> Milvus:
        return Milvus(
            embedding_function=embeddings,
            connection_args={"uri": self.connection_uri},
            collection_name=self.collection,
        )

    def _sync_get_or_create_vectorstore(self, main_loop: asyncio.AbstractEventLoop) -> Optional[Milvus]:
        connections.connect(uri=self.connection_uri)
        embeddings = self.get_embeddings()
        state = self._store_state()
        if self._stores_ready(state):
            logger.info("All stores already exist, reusing them.")
            return self._open_milvus(embeddings)
        docs = self.load_documents()
        if not docs:
            return self._handle_missing_documents(state, embeddings)
        chunks = split_documents(docs)
        self._sync_ensure_mongodb(docs)
        return self._initialize_stores(state, docs, chunks, embeddings, main_loop)

    def _stores_ready(self, state: StoreState) -> bool:
        return (
            state.has_milvus
            and state.has_graph
            and state.has_es_docs
            and not state.has_pending
            and not state.has_incomplete
        )

    def _handle_missing_documents(self, state: StoreState, embeddings) -> Milvus:
        if not state.has_milvus:
            raise ValueError(
                f"No documents found in {self.data_path}; cannot initialize the vectorstore"
            )
        logger.warning(
            f"No documents found in {self.data_path}, but Milvus collection exists. Returning existing collection."
        )
        return self._open_milvus(embeddings)

    def _initialize_stores(
        self,
        state: StoreState,
        docs: List[Document],
        chunks: List[Document],
        embeddings,
        main_loop: asyncio.AbstractEventLoop,
    ) -> Optional[Milvus]:
        vectorstore: Optional[Milvus] = None
        if not state.has_milvus:
            logger.info(f"Creating Milvus collection '{self.collection}' and indexing documents...")
            vectorstore = self.build_vectorstore(docs)
        else:
            logger.info(f"Milvus collection '{self.collection}' exists, checking for missing chunks...")
            vectorstore = self._sync_add_missing_chunks(self._open_milvus(embeddings), chunks)
        self._index_elasticsearch(state, chunks)
        self._schedule_graph_extraction(state, chunks, main_loop)
        return vectorstore

    def _index_elasticsearch(self, state: StoreState, chunks: List[Document]) -> None:
        if not (state.needs_es and self.es_retriever):
            return
        logger.info("Indexing documents to Elasticsearch...")
        self.es_retriever.index_documents(chunks)

    def _schedule_graph_extraction(self, state: StoreState, chunks: List[Document], main_loop) -> None:
        if not state.needs_graph:
            return
        for doc_id, doc_chunks in self._pending_graph_chunks(chunks).items():
            logger.info("Scheduling graph extraction in background...")
            asyncio.run_coroutine_threadsafe(self._async_extract_graph(doc_chunks, doc_id), main_loop)

    def _store_state(self) -> StoreState:
        has_es_docs = self._es_has_documents()
        has_mongo_docs = self.mongo_repo.documents.count_documents({}) > 0
        has_graph = not self.neo4j_repo.is_empty()
        has_incomplete = self.mongo_repo.has_incomplete_documents()
        return StoreState(
            has_milvus=utility.has_collection(self.collection),
            has_mongo_docs=has_mongo_docs,
            has_graph=has_graph,
            has_pending=bool(self.mongo_repo.get_pending_chunks(limit=1)),
            has_incomplete=has_incomplete,
            has_es_docs=has_es_docs,
            needs_es=self._needs_es(has_mongo_docs, has_es_docs, has_incomplete),
            needs_graph=has_mongo_docs and (not has_graph or has_incomplete),
        )

    def _needs_es(self, has_mongo_docs: bool, has_es_docs: bool, has_incomplete: bool) -> bool:
        return (
            self.use_hybrid_search
            and self.es_retriever
            and has_mongo_docs
            and (not has_es_docs or has_incomplete)
        )

    def _es_has_documents(self) -> bool:
        if not (self.es_retriever and self.es_retriever.es_client):
            return False
        try:
            return self.es_retriever.es_client.indices.exists(index=self.es_retriever.index_name)
        except Exception as error:
            logger.warning("ES index existence check failed", extra={"error": str(error)})
            return False
