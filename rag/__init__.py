"""RAG ingestion and query-answering package.

The former monolithic ``rag.py`` is split by responsibility:
``splitting`` (chunk strategies), ``document_loader`` (format dispatch),
``document_ingestion`` (MongoDB-backed incremental ingestion),
``vectorstore_support`` (Milvus/ES/graph synchronization) and
``local_rag`` (retrieval chain + query orchestration facade).
"""

from rag.document_loader import load_documents
from rag.local_rag import LocalRAG, command_line_rag
from rag.splitting import AdaptiveTextSplitter, select_text_splitter

__all__ = [
    "LocalRAG",
    "command_line_rag",
    "AdaptiveTextSplitter",
    "select_text_splitter",
    "load_documents",
]
