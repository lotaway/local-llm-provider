from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging

from globals import local_rag
from model_providers import LocalLLModel
from schemas import DocumentCheckRequest, ImportDocumentRequest
from rag import LocalRAG

router = APIRouter(prefix="/rag", tags=["rag"])
logger = logging.getLogger(__name__)


@router.get("/document/check")
async def check_document(req: DocumentCheckRequest):
    global local_rag

    if local_rag is None:
        local_model = LocalLLModel.init_local_model()
        local_rag = LocalRAG(local_model)

    exists = local_rag.check_document_exists(req.bvid, req.cid)
    return {"exists": exists}


@router.post("/document/import")
async def import_document(req: ImportDocumentRequest):
    global local_rag

    LocalLLModel.init_local_model()

    if local_rag is None:
        local_model = LocalLLModel.init_local_model()
        local_rag = LocalRAG(local_model)

    try:
        if local_rag.check_document_exists(req.bvid, req.cid):
            return {"data": None, "message": "Document already exists", "exists": True}

        result = local_rag.add_document(
            title=req.title,
            content=req.content,
            source=req.source,
            content_type=req.contentType,
            bvid=req.bvid,
            cid=req.cid,
        )
        return {"data": result, "exists": False}
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}
