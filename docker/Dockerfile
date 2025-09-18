# Multi-stage build for efficient Docker image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime as base

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    pip cache purge

# Copy source code
COPY src/ ./src/
COPY main.py .
COPY PROJECT_STRUCTURE.md .

# Create necessary directories
RUN mkdir -p data/raw data/processed checkpoints visualizations mlruns

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="file:/app/mlruns" \
    PYTHONPATH="/app" \
    CUDA_VISIBLE_DEVICES="0" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    NUMEXPR_MAX_THREADS=4 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('Container is healthy')" || exit 1

# Default command - run the complete pipeline
CMD ["python", "main.py"]