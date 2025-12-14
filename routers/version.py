from fastapi import APIRouter, Depends
from typing import Annotated

# Import business routers from controllers
from controllers.llm_controller import router as llm_router
from controllers.agent_controller import router as agent_router
from controllers.rag_controller import router as rag_router
from controllers.file_controller import router as file_router


def parse_version(version: str):
    """Dependency to validate and parse version parameter"""
    if version not in {"v1"}:
        raise ValueError("Unsupported version")
    return version


router = APIRouter(
    prefix="/{version}", dependencies=[Depends(parse_version)], tags=["version"]
)

router.include_router(llm_router)
router.include_router(agent_router)
router.include_router(rag_router)
router.include_router(file_router)
