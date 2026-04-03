from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from globals import limiter
import logging

router = APIRouter(prefix="/file", tags=["file"])
logger = logging.getLogger(__name__)


@router.post("/upload")
@limiter.limit("3/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    from utils import FileProcessor

    try:
        file_processor = FileProcessor()
        file_id, filename, _ = file_processor.save_uploaded_file(
            file.file, file.filename
        )

        return {
            "id": file_id,
            "filename": filename,
            "message": "File uploaded successfully",
        }
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
