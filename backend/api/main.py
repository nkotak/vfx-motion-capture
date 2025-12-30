"""
FastAPI main application for VFX Motion Capture.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.core.config import settings
from backend.core.exceptions import VFXException
from backend.api.routes import upload, generate, jobs, files, realtime
from backend.api.websocket import websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Ensure directories exist
    settings.ensure_directories()

    # Initialize services
    from backend.services.file_manager import get_file_manager
    from backend.services.job_manager import get_job_manager

    file_manager = get_file_manager()
    job_manager = get_job_manager()

    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())

    logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    logger.info("Shutdown complete")


async def periodic_cleanup():
    """Background task to clean up expired files and jobs."""
    from backend.services.file_manager import get_file_manager
    from backend.services.job_manager import get_job_manager

    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour

            file_manager = get_file_manager()
            job_manager = get_job_manager()

            files_cleaned = await file_manager.cleanup_expired()
            jobs_cleaned = await job_manager.cleanup_old_jobs()

            if files_cleaned or jobs_cleaned:
                logger.info(f"Cleanup: {files_cleaned} files, {jobs_cleaned} jobs")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Real-time VFX motion capture using AI video generation models",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(VFXException)
async def vfx_exception_handler(request: Request, exc: VFXException):
    """Handle custom VFX exceptions."""
    return JSONResponse(
        status_code=400,
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
        },
    )


# Include routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(generate.router, prefix="/api", tags=["Generate"])
app.include_router(jobs.router, prefix="/api", tags=["Jobs"])
app.include_router(files.router, prefix="/api", tags=["Files"])
app.include_router(realtime.router, prefix="/api", tags=["Real-time"])
app.include_router(websocket_router, tags=["WebSocket"])


# Mount static files for outputs
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    from backend.services.model_manager import get_model_manager
    
    # Simple check if model manager is initialized
    model_manager_status = "active"
    try:
        get_model_manager()
    except Exception as e:
        model_manager_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "version": settings.app_version,
        "inference_engine": model_manager_status,
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )
