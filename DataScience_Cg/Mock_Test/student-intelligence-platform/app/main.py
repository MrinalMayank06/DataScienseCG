from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.database import Base, engine
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting_application", extra={"env": settings.app_env})
    Base.metadata.create_all(bind=engine)
    yield
    logger.info("shutting_down_application")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
    }
