# Backend Documentation 

## Architecture

The backend is built with FastAPI and provides a robust API for road object detection. It uses state-of-the-art deep learning models including Faster R-CNN and YOLO for object detection.

## Core Components

### API Server (`api_server.py`)

#### Endpoints
```
POST /detect_objects/     - Detect objects in images
POST /segment_image/      - Perform image segmentation
POST /process_video/      - Process video files
GET  /health             - Health check endpoint
```

### Model Factory (`model_factory.py`)

Handles model initialization and management:
- Model loading and caching
- Weight management
- Device optimization (CPU/GPU)

### Detection Pipeline

1. **Input Processing**
   - Image validation and preprocessing
   - Video frame extraction
   - Format standardization

2. **Model Processing**
   - Object detection using selected model
   - Confidence scoring
   - Bounding box calculation

3. **Output Generation**
   - JSON response formatting
   - Image/video output generation
   - Performance metrics

## Models

### Faster R-CNN
- Based on ResNet50 backbone
- Pre-trained on COCO dataset
- Optimized for road object detection

### YOLO Integration
- Optional YOLO model support
- Custom weight loading
- Configurable detection parameters

## Object Classes

### Road-specific Classes
```python
ROAD_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter'
]
```

### Extended Classes
- Standard COCO classes
- Additional road infrastructure classes
- Custom class support

## Performance Optimization

1. **GPU Acceleration**
   - CUDA support
   - Batch processing
   - Memory management

2. **Video Processing**
   - Frame skipping
   - Resolution adaptation
   - Multi-threading

3. **API Performance**
   - Async processing
   - Connection pooling
   - Response caching

## Development Guide

### Adding New Models

1. Create model class in `model_factory.py`
2. Implement required interfaces
3. Add model configuration
4. Update documentation

### Error Handling

The backend implements comprehensive error handling:
- Input validation
- Model exceptions
- Resource management
- API errors

### Logging

Structured logging for:
- API requests
- Model performance
- Error tracking
- Resource usage

### Testing

```bash
# Run all tests
pytest

# Test specific components
pytest tests/test_api_server.py
pytest tests/test_model_factory.py
```

## Configuration

### Environment Variables
```bash
CUDA_VISIBLE_DEVICES=0,1  # GPU configuration
MODEL_CACHE_DIR=./cache   # Model storage location
LOG_LEVEL=INFO           # Logging configuration
```

### Model Configuration
```python
model_config = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.3,
    "device": "cuda"
}
```

## Deployment

### Requirements
- Python 3.12+
- PyTorch with CUDA support
- FastAPI
- OpenCV

### Docker Support
```bash
# Build container
docker build -t road-detection-backend .

# Run container
docker run -p 8000:8000 road-detection-backend
```

## Troubleshooting

### Common Issues

1. **Model Loading**
- Verify CUDA availability
- Check model cache directory
- Validate model weights

2. **Memory Issues**
- Monitor GPU memory usage
- Adjust batch sizes
- Check for memory leaks

3. **Performance**
- Profile API endpoints
- Monitor model inference time
- Check system resources

### Monitoring

The backend provides monitoring endpoints:
- `/health` for system status
- Performance metrics
- Resource utilization

## Security

- Input validation
- Rate limiting
- CORS configuration
- Authentication support