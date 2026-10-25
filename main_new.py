"""
主入口：新架构启动
"""
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bootstrap.container import initialize_container, shutdown_container, get_container
from interfaces.http.routes import chat_routes
from infrastructure.config.settings import get_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    settings = get_settings()
    logger.info(f"Starting server on {settings.server.host}:{settings.server.port}")
    
    container = await initialize_container(settings)
    
    # 可选：预加载默认模型
    # if settings.default_model:
    #     try:
    #         await container.load_model_use_case.execute(LoadModelCommand(...))
    #     except Exception as e:
    #         logger.warning(f"Failed to preload model: {e}")
    
    yield
    
    # 关闭
    logger.info("Shutting down server...")
    await shutdown_container()


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    settings = get_settings()
    
    app = FastAPI(
        title="Local LLM Provider",
        description="Local LLM inference server with RAG and multi-engine support",
        version="0.2.0",
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 路由
    app.include_router(chat_routes.router)
    
    # 根路径
    @app.get("/")
    async def root():
        return {
            "service": "Local LLM Provider",
            "version": "0.2.0",
            "architecture": "DDD + Clean Architecture",
        }
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        log_level=settings.server.log_level,
        reload=False,
    )