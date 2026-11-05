import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from domain.rag.ports import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/rag", tags=["rag"])

_rag_service: Optional[RAGService] = None


class IngestRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_id: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=100)


class GenerateRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=100)


def set_rag_service(rag_service: RAGService) -> None:
    global _rag_service
    _rag_service = rag_service


def _require_service() -> RAGService:
    if _rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return _rag_service


def _document_payload(document):
    return {
        "id": document.id,
        "content": document.content,
        "metadata": document.metadata,
        "score": document.score,
    }


def _retrieval_payload(result):
    return {
        "query": result.query,
        "total_count": result.total_count,
        "retrieval_time_ms": result.retrieval_time_ms,
        "documents": [_document_payload(item) for item in result.documents],
    }


@router.post("/documents")
async def ingest_document(request: IngestRequest):
    service = _require_service()
    doc_id = await service.ingest(request.content, request.metadata, request.doc_id)
    return {"doc_id": doc_id}


@router.post("/retrieve")
async def retrieve_documents(request: RetrieveRequest):
    service = _require_service()
    result = await service.retrieve(request.query, request.k)
    return _retrieval_payload(result)


@router.post("/generate")
async def generate_answer(request: GenerateRequest):
    service = _require_service()
    result = await service.generate(request.query, request.k, False)
    return {
        "answer": result.answer,
        "tokens_used": result.tokens_used,
        "generation_time_ms": result.generation_time_ms,
        "sources": [_document_payload(item) for item in result.sources],
    }


@router.get("/health")
async def rag_health():
    service = _require_service()
    return {"status": "healthy" if await service.health_check() else "degraded"}
