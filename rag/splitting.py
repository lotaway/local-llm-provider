"""Document chunking strategies for the RAG ingestion pipeline."""

import hashlib
import os
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

CODE_EXTENSIONS = (".py", ".java", ".kt", ".rs", ".js", ".ts", ".html", ".css", ".cs", ".swift")
STRUCTURED_EXTENSIONS = (".json", ".csv", ".xlsx", ".xls")

MARKDOWN_SEPARATORS = (
    "\n## ", "\n# ", "\n\n", "\n", "。", "！", "？", ".", "!", "?", ";", "；", ",", "，", " ",
)
CODE_SEPARATORS = ("\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ")
STRUCTURED_SEPARATORS = ("\n\n", "\n", ",", " ")


def document_checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def document_id(content: str) -> str:
    return f"doc_{document_checksum(content)[:16]}"


def chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}_chunk_{index:04d}"


class AdaptiveTextSplitter(TextSplitter):
    def __init__(self, min_chunk=500, max_chunk=2000, chunk_overlap=0, **kwargs):
        super().__init__(**kwargs)
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        if not chunks:
            return [text] if text else []
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk:
                self._append_oversized(final_chunks, chunk)
            elif len(chunk) < self.min_chunk:
                self._append_small(final_chunks, chunk)
            else:
                self._append_middle(final_chunks, chunk)
        return final_chunks

    def _append_oversized(self, final_chunks: List[str], chunk: str) -> None:
        current_temp = ""
        for sentence in re.split(r"(?<=[.!?。！？])\s+", chunk):
            if not sentence.strip():
                continue
            if len(current_temp) + len(sentence) <= self.max_chunk:
                current_temp += sentence + " "
                continue
            if current_temp:
                final_chunks.append(current_temp.strip())
            current_temp = self._overlap(final_chunks) + sentence + " "
        if current_temp:
            final_chunks.append(current_temp.strip())

    def _append_small(self, final_chunks: List[str], chunk: str) -> None:
        if final_chunks and len(final_chunks[-1]) + len(chunk) + 2 <= self.max_chunk:
            final_chunks[-1] += "\n\n" + chunk
        else:
            final_chunks.append(self._overlap(final_chunks) + chunk)

    def _append_middle(self, final_chunks: List[str], chunk: str) -> None:
        final_chunks.append(self._overlap(final_chunks) + chunk)

    def _overlap(self, final_chunks: List[str]) -> str:
        if self.chunk_overlap > 0 and final_chunks:
            return final_chunks[-1][-self.chunk_overlap :]
        return ""


def select_text_splitter(source: Optional[str], content_type: Optional[str]) -> TextSplitter:
    file_ext = ""
    if content_type:
        file_ext = f".{content_type.lstrip('.')}".lower()
    elif source:
        file_ext = os.path.splitext(source)[1].lower()

    if file_ext in (".md", ".markdown"):
        return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, separators=list(MARKDOWN_SEPARATORS))
    if file_ext in CODE_EXTENSIONS:
        return RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200, separators=list(CODE_SEPARATORS))
    if file_ext in STRUCTURED_EXTENSIONS:
        return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100, separators=list(STRUCTURED_SEPARATORS))
    return AdaptiveTextSplitter(min_chunk=500, max_chunk=2000, chunk_overlap=200)


def split_documents(docs: List[Document]) -> List[Document]:
    all_chunks: List[Document] = []
    for doc in docs:
        doc_id = document_id(doc.page_content)
        splitter = select_text_splitter(doc.metadata.get("source"), doc.metadata.get("content_type"))
        for index, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata.update({"chunk_id": chunk_id(doc_id, index), "doc_id": doc_id})
            all_chunks.append(chunk)
    return all_chunks
