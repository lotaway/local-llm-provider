"""MongoDB-backed incremental ingestion helpers consumed by ``LocalRAG``."""

import inspect
import logging
import os
from typing import Dict, List

from langchain_core.documents import Document
from pymilvus import utility
from langchain_milvus import Milvus

from constants import DB_COLLECTION
from rag.splitting import (
    chunk_id,
    document_checksum,
    document_id,
    select_text_splitter,
    split_documents,
)
from services.ingestion_checkpoint import (
    checksum_matches,
    is_complete,
    needs_rebuild,
    normalize_status,
    pending_steps,
)

logger = logging.getLogger(__name__)


class DocumentIngestionMixin:
    """Compound document add/checksum/rebuild/checkpoint behaviour for LocalRAG."""

    async def add_document(
        self, title: str, content: str, source: str, content_type: str = "md", **kwargs
    ):
        doc_id, file_path, filename, checksum = self._persist_source_file(title, content, source, content_type)
        existing = self.mongo_repo.get_document_by_path(file_path)
        if existing and checksum_matches(existing, checksum) and is_complete(existing.get("status")):
            return self._skipped_result(existing, filename)
        if needs_rebuild(existing, checksum):
            await self._purge_document(existing)
        self._save_document_record(doc_id, title, content, source, filename, file_path, content_type, checksum, kwargs)
        status = normalize_status(self.mongo_repo.get_document(doc_id).get("status"))
        chunks = await self._run_ingestion_steps(doc_id, content, source, title, content_type, status)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "pending_at_start": pending_steps(status),
        }

    async def _run_ingestion_steps(self, doc_id, content, source, title, content_type, status) -> List[Document]:
        chunks = await self._execute_step(
            doc_id, "chunked", lambda: self._ensure_chunks(doc_id, content, source, title, content_type, status)
        )
        await self._execute_step(doc_id, "embedded", lambda: self._ensure_embeddings(doc_id, chunks, status))
        await self._execute_step(doc_id, "graphed", lambda: self._ensure_graph(doc_id, chunks, status))
        return chunks

    def _persist_source_file(self, title: str, content: str, source: str, content_type: str) -> tuple:
        filename = os.path.basename(source) if source else f"{title}.{content_type}"
        file_path = os.path.join(self.data_path, "uploads", filename)
        os.makedirs(os.path.join(self.data_path, "uploads"), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return document_id(content), file_path, filename, document_checksum(content)

    def _skipped_result(self, document: dict, filename: str) -> dict:
        doc_id = document["_id"]
        chunks = self.mongo_repo.get_chunks_by_doc(doc_id)
        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks), "skipped": True}

    async def _execute_step(self, doc_id: str, step: str, operation):
        try:
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            self.mongo_repo.clear_document_error(doc_id)
            return result
        except Exception as error:
            self.mongo_repo.mark_document_error(doc_id, step, str(error))
            logger.error(
                f"Ingestion step failed: {step}",
                extra={"doc_id": doc_id, "step": step},
                exc_info=True,
            )
            raise

    def _save_document_record(
        self, doc_id, title, content, source, filename, file_path, content_type, checksum, kwargs
    ) -> None:
        metadata = kwargs.get("metadata", {})
        metadata.update({key: value for key, value in kwargs.items() if value is not None})
        self.mongo_repo.save_document(
            doc_id=doc_id,
            content=content,
            title=title or "Untitled",
            source=source,
            author=kwargs.get("author", ""),
            summary=kwargs.get("summary", ""),
            path=file_path,
            filename=filename,
            format=content_type,
            checksum=f"sha256:{checksum}",
            metadata=metadata,
        )

    async def _purge_document(self, document: dict) -> None:
        doc_id = document["_id"]
        chunk_ids = [chunk["chunk_id"] for chunk in self.mongo_repo.get_chunks_by_doc(doc_id)]
        self._delete_vectors(chunk_ids)
        if self.es_retriever:
            self.es_retriever.delete_by_chunk_ids(chunk_ids)
        self.neo4j_repo.delete_document_graph(doc_id)
        self.mongo_repo.delete_document(doc_id)
        logger.info(
            "Rebuilding document after checksum change",
            extra={"doc_id": doc_id, "chunk_count": len(chunk_ids)},
        )

    def _delete_vectors(self, chunk_ids: List[str]) -> None:
        if not chunk_ids or not utility.has_collection(self.collection):
            return
        vectorstore = Milvus(
            embedding_function=self.get_embeddings(),
            connection_args={"uri": self.connection_uri},
            collection_name=self.collection,
        )
        vectorstore.delete(ids=chunk_ids)

    def _ensure_chunks(self, doc_id, content, source, title, content_type, status) -> List[Document]:
        if status.get("chunked"):
            stored = self.mongo_repo.get_chunks_by_doc(doc_id)
            if stored:
                return [self._chunk_document(record) for record in stored]
        doc = Document(
            page_content=content,
            metadata={"source": source, "title": title, "doc_id": doc_id},
        )
        chunks = select_text_splitter(source, content_type).split_documents([doc])
        for index, chunk in enumerate(chunks):
            self._save_chunk(doc_id, index, chunk)
        self.mongo_repo.update_document_status(doc_id, "chunked", True)
        return chunks

    def _save_chunk(self, doc_id: str, index: int, chunk: Document) -> None:
        cid = chunk_id(doc_id, index)
        chunk.metadata.update({"chunk_id": cid, "doc_id": doc_id})
        self.mongo_repo.save_chunk(
            chunk_id=cid,
            doc_id=doc_id,
            index=index,
            text=chunk.page_content,
            offset_start=0,
            offset_end=len(chunk.page_content),
            metadata=chunk.metadata,
        )

    def _chunk_document(self, record: dict) -> Document:
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {"chunk_id": record["chunk_id"], "doc_id": record["doc_id"]}
        )
        return Document(page_content=record.get("text", ""), metadata=metadata)

    async def _ensure_embeddings(self, doc_id, chunks, status) -> None:
        if status.get("embedded"):
            return
        pending = [chunk for chunk in chunks if self._needs_embedding(chunk)]
        if pending:
            vectorstore = await self.get_or_create_vectorstore()
            if vectorstore is None:
                raise RuntimeError("no vector store available for embedding step")
            vectorstore.add_documents(pending)
            for chunk in pending:
                self.mongo_repo.update_chunk_embedding_status(chunk.metadata["chunk_id"], "done")
        if self.use_hybrid_search and self.es_retriever:
            self.es_retriever.index_documents(chunks)
        self.mongo_repo.update_document_status(doc_id, "embedded", True)

    def _needs_embedding(self, chunk: Document) -> bool:
        record = self.mongo_repo.get_chunk(chunk.metadata.get("chunk_id"))
        return bool(record) and record.get("embedding_status") != "done"

    async def _ensure_graph(self, doc_id, chunks, status) -> None:
        if status.get("graphed"):
            return
        succeeded = await self._async_extract_graph(chunks, doc_id)
        if succeeded is False:
            raise RuntimeError("graph extraction incomplete")

    async def _async_extract_graph(self, chunks: List[Document], source_doc_id: str):
        logger.info(f"Starting graph extraction for {len(chunks)} chunks...")
        if not chunks:
            return True
        doc_title = chunks[0].metadata.get("title", source_doc_id)
        self._register_document_node(source_doc_id, doc_title)
        counters = await self._extract_chunk_graphs(chunks, source_doc_id)
        return self._finish_graph_extraction(source_doc_id, counters)

    async def _extract_chunk_graphs(self, chunks: List[Document], source_doc_id: str) -> Dict[str, int]:
        counters = {"entities": 0, "relations": 0, "failed": 0}
        for index, chunk in enumerate(chunks):
            try:
                chunk_id_value, counts = await self._extract_chunk_graph(chunk, index, source_doc_id)
                counters["entities"] += counts["entities"]
                counters["relations"] += counts["relations"]
                logger.info(
                    f"Extracted {counts['entities']} entities and {counts['relations']} relations from chunk {index + 1}"
                )
            except Exception as error:
                counters["failed"] += 1
                logger.error(
                    f"Error during graph extraction for chunk {index + 1}: {error}",
                    extra={"doc_id": source_doc_id, "index": index},
                )
        return counters

    async def _extract_chunk_graph(self, chunk: Document, index: int, source_doc_id: str) -> tuple:
        cid = chunk.metadata.get("chunk_id", chunk_id(source_doc_id, index))
        self._register_chunk_pointer(cid, index, source_doc_id)
        entities, relations = await self.graph_extractor.extract_graph(chunk.page_content)
        for entity in entities:
            self.neo4j_repo.merge_entity(entity)
            self._mark_chunk_described(cid, entity.stable_id)
        for relation in relations:
            relation.source_doc_id = source_doc_id
            self.neo4j_repo.merge_relation(relation)
        logger.debug(f"Processing chunk {index + 1} (chunk_id: {cid})")
        return cid, {"entities": len(entities), "relations": len(relations)}

    def _finish_graph_extraction(self, source_doc_id: str, counters: Dict[str, int]) -> bool:
        total_entities = counters["entities"]
        total_relations = counters["relations"]
        logger.info(
            f"Graph extraction completed. Total entities: {total_entities}, Total relations: {total_relations}"
        )
        if counters["failed"]:
            self.mongo_repo.mark_document_error(
                source_doc_id, "graphed", f"{counters['failed']} chunks failed graph extraction"
            )
            return False
        self.mongo_repo.update_document_status(source_doc_id, "graphed", True)
        return True

    def _register_document_node(self, source_doc_id: str, doc_title: str) -> None:
        self.neo4j_repo.session.run(
            "MERGE (d:Document {doc_id: $doc_id}) SET d.filename = $filename",
            doc_id=source_doc_id,
            filename=doc_title,
        )

    def _register_chunk_pointer(self, cid: str, index: int, source_doc_id: str) -> None:
        self.neo4j_repo.session.run(
            "MERGE (c:Chunk {chunk_id: $chunk_id}) SET c.doc_id = $doc_id, c.index = $index",
            chunk_id=cid,
            doc_id=source_doc_id,
            index=index,
        )
        self.neo4j_repo.session.run(
            """
            MATCH (d:Document {doc_id: $doc_id})
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (d)-[:CONTAINS]->(c)
            """,
            doc_id=source_doc_id,
            chunk_id=cid,
        )

    def _mark_chunk_described(self, cid: str, stable_id: str) -> None:
        self.neo4j_repo.session.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MATCH (e:Entity {stable_id: $stable_id})
            MERGE (c)-[:DESCRIBES]->(e)
            """,
            chunk_id=cid,
            stable_id=stable_id,
        )

    def check_document_exists(self, bvid: str, cid: int) -> bool:
        if not self.es_retriever:
            return False
        try:
            query = {
                "bool": {
                    "must": [
                        {"match": {"metadata.bvid": bvid}},
                        {"match": {"metadata.cid": cid}},
                    ]
                }
            }
            res = self.es_retriever.es_client.search(
                index=self.es_retriever.index_name, query=query, size=1
            )
            return len(res["hits"]["hits"]) > 0
        except Exception as e:
            logger.error(f"Error checking document existence in ES: {e}")
            return False

    def _pending_graph_chunks(self, chunks: List[Document]) -> Dict[str, List[Document]]:
        grouped: Dict[str, List[Document]] = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id")
            if not doc_id:
                continue
            document = self.mongo_repo.get_document(doc_id)
            status = (document or {}).get("status", {})
            if not status.get("graphed", False):
                grouped.setdefault(doc_id, []).append(chunk)
        return grouped

    def _sync_ensure_mongodb(self, docs: List[Document]) -> None:
        for doc in docs:
            self._sync_reconcile_document(doc)

    def _sync_reconcile_document(self, doc: Document) -> None:
        checksum = document_checksum(doc.page_content)
        doc_id = document_id(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        existing = self.mongo_repo.get_document_by_path(source)
        if existing and checksum_matches(existing, checksum):
            self._sync_ensure_document_chunks(existing["_id"], doc)
            return
        if needs_rebuild(existing, checksum):
            logger.info(
                "Document checksum changed, rebuilding",
                extra={"doc_id": existing["_id"], "path": source},
            )
            self._sync_purge_document(existing)
        self._sync_save_document(doc, doc_id, checksum)
        self._sync_ensure_document_chunks(doc_id, doc)

    def _sync_save_document(self, doc: Document, doc_id: str, checksum: str) -> None:
        source = doc.metadata.get("source", "unknown")
        filename = os.path.basename(source) if source else "unknown"
        self.mongo_repo.save_document(
            doc_id=doc_id,
            content=doc.page_content,
            title=doc.metadata.get("title") or filename or "Untitled",
            source=source,
            author=doc.metadata.get("author", ""),
            summary=doc.metadata.get("summary", ""),
            path=source,
            filename=filename,
            format="auto",
            checksum=f"sha256:{checksum}",
            metadata=doc.metadata,
        )
        logger.info(f"Added new document {doc_id} to MongoDB")

    def _sync_ensure_document_chunks(self, doc_id: str, doc: Document) -> None:
        document = self.mongo_repo.get_document(doc_id) or {}
        if document.get("status", {}).get("chunked") and self.mongo_repo.get_chunks_by_doc(doc_id):
            return
        splitter = select_text_splitter(
            doc.metadata.get("source"), doc.metadata.get("content_type")
        )
        for index, chunk in enumerate(splitter.split_documents([doc])):
            if self.mongo_repo.get_chunk(chunk_id(doc_id, index)):
                continue
            self._save_chunk(doc_id, index, chunk)
        self.mongo_repo.update_document_status(doc_id, "chunked", True)

    def _sync_purge_document(self, document: dict) -> None:
        doc_id = document["_id"]
        chunk_ids = [chunk["chunk_id"] for chunk in self.mongo_repo.get_chunks_by_doc(doc_id)]
        self.mongo_repo.delete_document(doc_id)
        if chunk_ids and utility.has_collection(self.collection):
            vectorstore = Milvus(
                embedding_function=self.get_embeddings(),
                connection_args={"uri": self.connection_uri},
                collection_name=self.collection,
            )
            vectorstore.delete(ids=chunk_ids)
        if self.es_retriever:
            self.es_retriever.delete_by_chunk_ids(chunk_ids)
        self.neo4j_repo.delete_document_graph(doc_id)

    def _sync_add_missing_chunks(self, vectorstore: Milvus, all_chunks: List[Document]) -> Milvus:
        missing_chunks = self._missing_embedding_chunks(all_chunks)
        if missing_chunks:
            logger.info(f"Adding {len(missing_chunks)} missing chunks to Milvus...")
            self._embed_missing_chunks(vectorstore, missing_chunks)
        else:
            logger.info("No missing chunks to add to Milvus")
        self._mark_embedded_documents(all_chunks)
        return vectorstore

    def _missing_embedding_chunks(self, all_chunks: List[Document]) -> List[Document]:
        missing = []
        for chunk in all_chunks:
            chunk_id_value = chunk.metadata.get("chunk_id")
            if not chunk_id_value:
                continue
            mongo_chunk = self.mongo_repo.get_chunk(chunk_id_value)
            if mongo_chunk and mongo_chunk.get("embedding_status") != "done":
                missing.append(chunk)
        return missing

    def _embed_missing_chunks(self, vectorstore: Milvus, missing_chunks: List[Document]) -> None:
        batch_size = 50
        for start in range(0, len(missing_chunks), batch_size):
            batch = missing_chunks[start : start + batch_size]
            vectorstore.add_documents(batch)
            for chunk in batch:
                cid = chunk.metadata.get("chunk_id")
                if cid:
                    self.mongo_repo.update_chunk_embedding_status(cid, "done")

    def _mark_embedded_documents(self, all_chunks: List[Document]) -> None:
        doc_ids = {chunk.metadata.get("doc_id") for chunk in all_chunks}
        for doc_id in doc_ids:
            if not doc_id:
                continue
            chunks = self.mongo_repo.get_chunks_by_doc(doc_id)
            if chunks and all(chunk.get("embedding_status") == "done" for chunk in chunks):
                self.mongo_repo.update_document_status(doc_id, "embedded", True)

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        return split_documents(docs)
