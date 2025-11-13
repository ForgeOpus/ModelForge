"""
Refactored FastAPI application for ModelForge.
Clean architecture with dependency injection and proper error handling.
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import finetuning_router, models_router, playground_router, hub_management_router
from .exceptions import (
    ModelForgeException,
    ModelAccessError,
    DatasetValidationError,
    TrainingError,
    ConfigurationError,
    HardwareError,
    DatabaseError,
)
from .logging_config import logger, setup_logging
from .dependencies import reset_services


# Setup logging
setup_logging("INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("ModelForge starting up...")
    logger.info("=" * 80)

    try:
        # Initialize services (will be created on first request via DI)
        logger.info("Services will be initialized on first request")
        yield

    finally:
        # Shutdown
        logger.info("ModelForge shutting down...")
        reset_services()
        logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ModelForge",
    description="Modular fine-tuning platform with support for multiple providers and strategies",
    version="2.0.0",
    lifespan=lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers
@app.exception_handler(ModelAccessError)
async def model_access_error_handler(request: Request, exc: ModelAccessError):
    """Handle model access errors."""
    logger.error(f"Model access error: {exc}")
    return JSONResponse(
        status_code=403,
        content={
            "error": "ModelAccessError",
            "message": str(exc),
            "detail": "You do not have access to this model. Please check permissions.",
        }
    )


@app.exception_handler(DatasetValidationError)
async def dataset_validation_error_handler(request: Request, exc: DatasetValidationError):
    """Handle dataset validation errors."""
    logger.error(f"Dataset validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "DatasetValidationError",
            "message": str(exc),
            "detail": "Dataset does not meet requirements. Please check format and fields.",
        }
    )


@app.exception_handler(TrainingError)
async def training_error_handler(request: Request, exc: TrainingError):
    """Handle training errors."""
    logger.error(f"Training error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "TrainingError",
            "message": str(exc),
            "detail": "An error occurred during training.",
        }
    )


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    """Handle configuration errors."""
    logger.error(f"Configuration error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "ConfigurationError",
            "message": str(exc),
            "detail": "Invalid configuration. Please check your settings.",
        }
    )


@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError):
    """Handle database errors."""
    logger.error(f"Database error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "DatabaseError",
            "message": str(exc),
            "detail": "A database error occurred.",
        }
    )


@app.exception_handler(ModelForgeException)
async def modelforge_exception_handler(request: Request, exc: ModelForgeException):
    """Handle general ModelForge exceptions."""
    logger.error(f"ModelForge error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
        }
    )


# Include routers
app.include_router(finetuning_router.router, prefix="/api", tags=["Fine-tuning"])
app.include_router(models_router.router, prefix="/api", tags=["Models"])
app.include_router(playground_router.router, prefix="/api", tags=["Playground"])
app.include_router(hub_management_router.router, prefix="/api", tags=["Hub Management"])


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "message": "ModelForge is running",
    }


# Info endpoint
@app.get("/api/info")
async def get_info():
    """Get ModelForge information."""
    from .providers.provider_factory import ProviderFactory
    from .strategies.strategy_factory import StrategyFactory

    return {
        "name": "ModelForge",
        "version": "2.0.0",
        "description": "Modular fine-tuning platform",
        "available_providers": ProviderFactory.get_available_providers(),
        "available_strategies": StrategyFactory.get_available_strategies(),
        "supported_tasks": [
            "text-generation",
            "summarization",
            "extractive-question-answering",
        ],
    }


# Root endpoint
@app.get("/api")
async def root():
    """Root API endpoint."""
    return {
        "message": "Welcome to ModelForge API v2.0",
        "docs": "/docs",
        "health": "/api/health",
        "info": "/api/info",
    }


# Mount frontend static files (if available)
frontend_path = os.path.join(os.path.dirname(__file__), "Frontend", "build")
if os.path.exists(frontend_path):
    logger.info(f"Mounting frontend from: {frontend_path}")
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    logger.warning(f"Frontend not found at: {frontend_path}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_new:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
