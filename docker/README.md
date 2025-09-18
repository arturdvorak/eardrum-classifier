# Docker Setup for Eardrum Classification

This directory contains Docker configuration for running the eardrum classification project in a containerized environment.

## Files

- `Dockerfile` - Multi-stage Docker build configuration
- `.dockerignore` - Files to exclude from Docker build context
- `README.md` - This documentation

## Quick Start

### Build and Run with Docker Compose

```bash
# Build and run the complete pipeline
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Build and Run with Docker

```bash
# Build the image
docker build -f docker/Dockerfile -t eardrum-classifier .

# Run the complete pipeline
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/checkpoints:/app/checkpoints \
           -v $(pwd)/visualizations:/app/visualizations \
           -v $(pwd)/mlruns:/app/mlruns \
           eardrum-classifier

# Run specific steps
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/checkpoints:/app/checkpoints \
           -v $(pwd)/visualizations:/app/visualizations \
           -v $(pwd)/mlruns:/app/mlruns \
           eardrum-classifier python main.py --step data

docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/checkpoints:/app/checkpoints \
           -v $(pwd)/visualizations:/app/visualizations \
           -v $(pwd)/mlruns:/app/mlruns \
           eardrum-classifier python main.py --step train

docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/checkpoints:/app/checkpoints \
           -v $(pwd)/visualizations:/app/visualizations \
           -v $(pwd)/mlruns:/app/mlruns \
           eardrum-classifier python main.py --step evaluate
```

## MLflow UI

Access the MLflow UI at: http://localhost:5000

```bash
# Start MLflow UI
docker-compose up mlflow-ui
```

## Environment Variables

The following environment variables can be customized:

- `PYTHONUNBUFFERED=1` - Python output buffering
- `MLFLOW_TRACKING_URI=file:/app/mlruns` - MLflow tracking location
- `CUDA_VISIBLE_DEVICES=0` - GPU device selection
- `BATCH_SIZE=32` - Training batch size
- `IMAGE_SIZE=224` - Image input size
- `MAX_EPOCHS=20` - Maximum training epochs
- `LEARNING_RATE=0.001` - Learning rate

## Volumes

The following directories are mounted as volumes for data persistence:

- `./data` → `/app/data` - Dataset files
- `./checkpoints` → `/app/checkpoints` - Model checkpoints
- `./visualizations` → `/app/visualizations` - Generated plots
- `./mlruns` → `/app/mlruns` - MLflow experiment data

## GPU Support

To enable GPU support, uncomment the GPU configuration in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure Docker has access to mounted volumes
2. **CUDA errors**: Check if NVIDIA Docker runtime is installed
3. **Memory issues**: Increase Docker memory allocation
4. **Port conflicts**: Change port mappings in docker-compose.yml

### Debugging

```bash
# Run interactive shell
docker run -it --rm eardrum-classifier /bin/bash

# Check container logs
docker logs eardrum-classifier

# Inspect container
docker inspect eardrum-classifier
```
