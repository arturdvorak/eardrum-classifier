"""
FastAPI Inference Service

This module provides a REST API for eardrum classification using ONNX Runtime.
It includes health checks, single/batch prediction endpoints, and model management.
"""

import os
import io
import json
import logging
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

from .onnx_inference import ONNXInference, create_inference_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    processing_time_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None

class ModelInfoResponse(BaseModel):
    model_path: str
    input_shape: List[int]
    output_shape: List[int]
    providers: List[str]
    class_names: List[str]
    performance_stats: Optional[Dict[str, float]] = None

# Initialize FastAPI app
app = FastAPI(
    title="Eardrum Classification API",
    description="Medical AI API for eardrum infection classification using ONNX Runtime",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine: Optional[ONNXInference] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup."""
    global inference_engine
    
    try:
        # Get model path from environment or use default
        model_path = os.getenv("MODEL_PATH", "models/onnx/eardrum_classifier_full.onnx")
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize inference engine
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        inference_engine = create_inference_engine(model_path, use_gpu=use_gpu)
        
        logger.info("Inference engine initialized successfully")
        logger.info(f"Model: {model_path}")
        logger.info(f"GPU enabled: {use_gpu}")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Eardrum Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        if inference_engine is None:
            return HealthResponse(
                status="unhealthy",
                model_loaded=False
            )
        
        # Test model with dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        _ = inference_engine.session.run(None, {'input': dummy_input})
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_info=inference_engine.get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False
        )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed model information."""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model_info = inference_engine.get_model_info()
        
        # Convert dynamic shapes to static for API response
        input_shape = model_info.get('input_shape', [])
        output_shape = model_info.get('output_shape', [])
        
        # Replace dynamic batch_size with 1 for API response
        if isinstance(input_shape, list) and len(input_shape) > 0:
            input_shape = [1 if x == 'batch_size' else x for x in input_shape]
        if isinstance(output_shape, list) and len(output_shape) > 0:
            output_shape = [1 if x == 'batch_size' else x for x in output_shape]
        
        return ModelInfoResponse(
            model_path=model_info.get('model_path', ''),
            input_shape=input_shape,
            output_shape=output_shape,
            providers=model_info.get('providers', []),
            class_names=model_info.get('class_names', [])
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/benchmark")
async def benchmark_model():
    """Run model performance benchmark."""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        stats = inference_engine.benchmark(num_runs=100)
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(image: UploadFile = File(...)):
    """Predict eardrum condition from a single image."""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate file type - check both content_type and filename
        is_image = False
        if image.content_type and image.content_type.startswith('image/'):
            is_image = True
        elif image.filename:
            # Check file extension as fallback
            valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
            file_ext = image.filename.lower().split('.')[-1]
            if f'.{file_ext}' in valid_extensions:
                is_image = True
        
        if not is_image:
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Load image
        try:
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
        
        # Run prediction
        import time
        start_time = time.time()
        
        result = inference_engine.predict_single(pil_image)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictionResponse(
            success=True,
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(images: List[UploadFile] = File(...)):
    """Predict eardrum conditions from multiple images."""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if len(images) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
        
        # Process images
        pil_images = []
        for image in images:
            # Validate file type - check both content_type and filename
            is_image = False
            if image.content_type and image.content_type.startswith('image/'):
                is_image = True
            elif image.filename:
                # Check file extension as fallback
                valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
                file_ext = image.filename.lower().split('.')[-1]
                if f'.{file_ext}' in valid_extensions:
                    is_image = True
            
            if not is_image:
                raise HTTPException(status_code=400, detail="All files must be images")
            
            image_data = await image.read()
            try:
                pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                pil_images.append(pil_image)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
        
        # Run batch prediction
        import time
        start_time = time.time()
        
        results = inference_engine.predict_batch(pil_images)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return BatchPredictionResponse(
            success=True,
            predictions=results,
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/url")
async def predict_from_url(image_url: str):
    """Predict from image URL (for testing purposes)."""
    try:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Download image from URL
        import requests
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Load image
        pil_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Run prediction
        import time
        start_time = time.time()
        
        result = inference_engine.predict_single(pil_image)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            success=True,
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"URL prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": [
            "/", "/health", "/model/info", "/predict", "/predict/batch", "/predict/url"
        ]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
