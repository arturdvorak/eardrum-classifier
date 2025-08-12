# Use official PyTorch image with CUDA (or switch to CPU version if needed)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy all files from your local project into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="file:/app/mlruns"

# Set default command (can be overridden when running the container)
CMD ["python", "scripts/1_train.py"]