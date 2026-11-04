"""Format-dispatched document loading for the RAG ingestion pipeline."""

import json
import logging
import os
from typing import Callable, List, Optional

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

from file_loaders import ChatGPTLoader, DeepSeekLoader

logger = logging.getLogger(__name__)

AfterDocLoad = Callable[[List[Document], str], List[Document]]

CODE_EXTENSIONS = (".py", ".java", ".kt", ".rs", ".js", ".ts", ".html", ".css", ".cs", ".swift")


def identity_after_load(docs: List[Document], _path: str) -> List[Document]:
    return docs


def _load_text(file_path: str) -> List[Document]:
    return TextLoader(file_path, encoding="utf-8").load()


def _load_code(file_path: str) -> List[Document]:
    return TextLoader(file_path, encoding="utf-8").load()


def _load_markdown(file_path: str) -> List[Document]:
    try:
        return UnstructuredMarkdownLoader(file_path).load()
    except Exception as error:
        logger.warning(
            "Markdown loader failed, falling back to TextLoader",
            extra={"file": file_path, "error": str(error)},
        )
        return TextLoader(file_path, encoding="utf-8").load()


def _load_pdf(file_path: str) -> List[Document]:
    return PyPDFLoader(file_path).load()


def _load_word(file_path: str) -> List[Document]:
    return UnstructuredWordDocumentLoader(file_path).load()


def _load_presentation(file_path: str) -> List[Document]:
    return UnstructuredPowerPointLoader(file_path).load()


def _load_spreadsheet(file_path: str) -> List[Document]:
    return UnstructuredExcelLoader(file_path).load()


def _load_csv(file_path: str) -> List[Document]:
    return CSVLoader(file_path).load()


def _detect_chat_export(data, file_path: str) -> Optional[List[Document]]:
    for loader in (ChatGPTLoader(), DeepSeekLoader()):
        if loader.check(data):
            logger.info("Detected chat export format", extra={"file": file_path})
            return loader.load(data, file_path)
    return None


def _load_json(file_path: str) -> List[Document]:
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        chat_docs = _detect_chat_export(data, file_path)
        if chat_docs is not None:
            return chat_docs
    except Exception as error:
        logger.warning(
            "Chat export parsing failed, falling back to JSONLoader",
            extra={"file": file_path, "error": str(error)},
        )
    return JSONLoader(file_path=file_path, jq_schema=".", text_content=True).load()


EXTENSION_LOADERS = {
    ".txt": _load_text,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".pdf": _load_pdf,
    ".docx": _load_word,
    ".doc": _load_word,
    ".pptx": _load_presentation,
    ".ppt": _load_presentation,
    ".xlsx": _load_spreadsheet,
    ".xls": _load_spreadsheet,
    ".csv": _load_csv,
    ".json": _load_json,
    **{ext: _load_code for ext in CODE_EXTENSIONS},
}


def load_documents(data_path: str, after_doc_load: AfterDocLoad = identity_after_load) -> List[Document]:
    """Walk ``data_path`` and load every supported file into LangChain documents."""
    docs: List[Document] = []
    for root, _, files in os.walk(data_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            loader = EXTENSION_LOADERS.get(file_ext)
            if loader is None:
                logger.warning("Skipping unsupported file format", extra={"file": file})
                continue
            docs.extend(_load_file(loader, os.path.join(root, file), file, after_doc_load))
    return docs


def _load_file(loader: Callable[[str], List[Document]], file_path: str, name: str, after_doc_load: AfterDocLoad) -> List[Document]:
    try:
        loaded = loader(file_path)
        return after_doc_load(loaded, name)
    except Exception as error:
        logger.error("Loading file failed", extra={"file": file_path, "error": str(error)}, exc_info=True)
        return []
