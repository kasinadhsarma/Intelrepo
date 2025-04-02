# Road Object Detection System

## Table of Contents

1. [System Overview](#system-overview)
   - [Introduction](#introduction)
   - [Key Features](#key-features)
   - [System Architecture](#system-architecture)
   - [Use Cases](#use-cases)

2. [Installation Guide](#installation-guide)
   - [Prerequisites](#prerequisites)
   - [Backend Setup](#backend-setup)
   - [Frontend Setup](#frontend-setup)
   - [Model Setup](#model-setup)
   - [Configuration](#configuration)
   - [Troubleshooting](#troubleshooting)

3. [Frontend Documentation](#frontend-documentation)
   - [Overview](#frontend-overview)
   - [Core Components](#core-components)
   - [UI Components](#ui-components)
   - [State Management](#state-management)
   - [API Integration](#api-integration)
   - [Performance Optimizations](#performance-optimizations)
   - [Directory Structure](#directory-structure)
   - [Development Guide](#frontend-development-guide)
   - [Troubleshooting](#frontend-troubleshooting)

4. [Backend Documentation](#backend-documentation)
   - [Architecture](#backend-architecture)
   - [Core Components](#backend-core-components)
   - [Models](#models)
   - [Object Classes](#object-classes)
   - [Performance Optimization](#performance-optimization)
   - [Development Guide](#backend-development-guide)
   - [Configuration](#backend-configuration)
   - [Deployment](#backend-deployment)
   - [Troubleshooting](#backend-troubleshooting)
   - [Security](#security)

5. [API Reference](#api-reference)
   - [Base URL](#base-url)
   - [Endpoints](#endpoints)
   - [Error Codes](#error-codes)
   - [Rate Limiting](#rate-limiting)
   - [Authentication](#authentication)

6. [Developer Guide](#developer-guide)
   - [Development Environment Setup](#development-environment-setup)
   - [Project Structure](#project-structure)
   - [Development Workflow](#development-workflow)
   - [Testing](#testing)
   - [Code Style](#code-style)
   - [Adding New Features](#adding-new-features)
   - [Model Development](#model-development)
   - [Performance Guidelines](#performance-guidelines)
   - [Security Best Practices](#security-best-practices)
   - [Deployment](#deployment)
   - [Monitoring and Logging](#monitoring-and-logging)
   - [Contributing Guidelines](#contributing-guidelines)
   - [Support](#support)

## System Overview

### Introduction

The Road Object Detection system is a comprehensive solution for analyzing road scenes using state-of-the-art deep learning models. It provides real-time object detection, image processing, and video analysis capabilities.

### Key Features

- Real-time object detection
- Support for multiple deep learning models (Faster R-CNN and YOLO)
- Image, video, and live camera processing
- Interactive web interface
- RESTful API for integration
- Performance optimization for resource-efficient processing

### System Architecture

#### Frontend Layer
- Built with Next.js 15.2
- Responsive UI using Tailwind CSS
- Real-time video processing capabilities
- Component-based architecture

#### Backend Layer
- FastAPI-based Python server
- Multiple model support:
  - Faster R-CNN
  - YOLO
- Asynchronous processing
- RESTful API endpoints

#### Detection Pipeline
1. Input Processing
   - Image/Video frame capture
   - Pre-processing and normalization
   
2. Model Inference
   - Object detection using selected model
   - Confidence scoring
   - Bounding box generation

3. Post-Processing
   - Result filtering
   - Visualization
   - Performance metrics calculation

### Use Cases

1. Traffic Monitoring
   - Vehicle detection and counting
   - Traffic flow analysis
   - Safety monitoring

2. Road Safety Analysis
   - Pedestrian detection
   - Traffic sign recognition
   - Hazard identification

3. Infrastructure Inspection
   - Road condition assessment
   - Traffic signal monitoring
   - Construction zone analysis

## Installation Guide

### Prerequisites

- Python 3.12+
- Node.js (Latest LTS version)
- CUDA-capable GPU (recommended)
- Git

### Backend Setup

1. Clone the repository:
```bash
git repo clone https://github.com/kasinadhsarma/Intelrepo
cd Intelrepo
```

2. Set up Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
cd backend
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
npm install
# or
pnpm install
```

2. Run the development server:
```bash
npm run dev
# or
pnpm dev
```

The application will be available at `http://localhost:3000`

### Model Setup

The system uses pre-trained models that will be downloaded automatically on first use:
- Faster R-CNN (ResNet50 backbone)
- YOLO (if configured)

### Configuration

#### Backend Configuration
- Edit `backend/api_server.py` for API settings
- Modify model parameters in `backend/model_factory.py`

#### Frontend Configuration
- Environment variables can be set in `.env.local`
- API endpoint configuration in `lib/api-client.ts`

### Troubleshooting

#### Common Issues

1. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. Port Conflicts
```bash
# Check if ports are in use
netstat -ano | findstr "8000"  # Windows
lsof -i :8000                  # Linux/Mac
```

3. Model Download Issues
- Ensure internet connectivity
- Check write permissions in the model cache directory

#### Getting Help

For additional help:
- Open an issue on GitHub
- Contact support at exploit0xffff@gmail.com

## Frontend Documentation

### Frontend Overview

The frontend is built with Next.js 15.2 and provides a modern, responsive interface for road object detection tasks.

### Core Components

1. **ImageCapture**
- Handles single image uploads
- Supports drag-and-drop functionality
- Displays detection results with bounding boxes
- Shows AI-generated scene descriptions

2. **VideoCapture**
- Processes video files
- Shows progress tracking
- Performs frame-by-frame analysis
- Provides exportable results

3. **LiveCapture**
- Real-time camera feed processing
- FPS counter display
- Live object detection visualization
- Adjustable detection settings

### UI Components

Built using a comprehensive UI kit with Radix UI primitives:
- Cards for structured content display
- Dialogs for user interactions
- Tooltips for enhanced usability
- Progress indicators for operations
- Alert components for notifications
- Custom buttons and form inputs

### State Management

- Context-based state management
- Efficient render optimization
- Real-time updates handling

### API Integration

The frontend integrates with the backend API through:
- Dedicated API client (`api-client.ts`)
- File upload handling
- Streaming response processing
- Error state management

### Performance Optimizations

1. **Image Processing**
- Client-side image compression
- Efficient canvas rendering
- WebGL acceleration when available

2. **Video Handling**
- Chunked upload support
- Stream processing
- Frame rate optimization

3. **Real-time Detection**
- Frame skipping for performance
- Resolution adaptation
- WebWorker processing

### Directory Structure

```
components/
├── image-capture.tsx      # Image processing component
├── live-capture.tsx       # Real-time camera component
├── video-capture.tsx      # Video processing component
├── theme-provider.tsx     # Theme management
└── ui/                    # Reusable UI components
    ├── button.tsx
    ├── card.tsx
    └── ...

lib/
├── api-client.ts         # API integration
└── utils.ts              # Utility functions
```

### Frontend Development Guide

#### Adding New Features

1. Create new components in `components/`
2. Update API client if needed
3. Add tests for new functionality
4. Update documentation

#### Code Style

- Follow TypeScript best practices
- Use functional components
- Implement proper error handling
- Add JSDoc comments for complex logic

#### Testing

Run tests using:
```bash
npm test
# or
pnpm test
```

### Frontend Troubleshooting

Common frontend issues and solutions:

1. **Performance Issues**
- Check browser console for warnings
- Monitor memory usage
- Verify WebGL support

2. **API Connection**
- Verify API endpoint configuration
- Check CORS settings
- Monitor network requests

3. **Camera Access**
- Ensure proper permissions
- Check SSL configuration
- Verify browser compatibility

## Backend Documentation 

### Backend Architecture

The backend is built with FastAPI and provides a robust API for road object detection. It uses state-of-the-art deep learning models including Faster R-CNN and YOLO for object detection.

### Backend Core Components

#### API Server (`api_server.py`)

##### Endpoints
```
POST /detect_objects/     - Detect objects in images
POST /segment_image/      - Perform image segmentation
POST /process_video/      - Process video files
GET  /health             - Health check endpoint
```

#### Model Factory (`model_factory.py`)

Handles model initialization and management:
- Model loading and caching
- Weight management
- Device optimization (CPU/GPU)

#### Detection Pipeline

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

### Models

#### Faster R-CNN
- Based on ResNet50 backbone
- Pre-trained on COCO dataset
- Optimized for road object detection

#### YOLO Integration
- Optional YOLO model support
- Custom weight loading
- Configurable detection parameters

### Object Classes

#### Road-specific Classes
```python
ROAD_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter'
]
```

#### Extended Classes
- Standard COCO classes
- Additional road infrastructure classes
- Custom class support

### Performance Optimization

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

### Backend Development Guide

#### Adding New Models

1. Create model class in `model_factory.py`
2. Implement required interfaces
3. Add model configuration
4. Update documentation

#### Error Handling

The backend implements comprehensive error handling:
- Input validation
- Model exceptions
- Resource management
- API errors

#### Logging

Structured logging for:
- API requests
- Model performance
- Error tracking
- Resource usage

#### Testing

```bash
# Run all tests
pytest

# Test specific components
pytest tests/test_api_server.py
pytest tests/test_model_factory.py
```

### Backend Configuration

#### Environment Variables
```bash
CUDA_VISIBLE_DEVICES=0,1  # GPU configuration
MODEL_CACHE_DIR=./cache   # Model storage location
LOG_LEVEL=INFO           # Logging configuration
```

#### Model Configuration
```python
model_config = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.3,
    "device": "cuda"
}
```

### Backend Deployment

#### Requirements
- Python 3.12+
- PyTorch with CUDA support
- FastAPI
- OpenCV

#### Docker Support
```bash
# Build container
docker build -t road-detection-backend .

# Run container
docker run -p 8000:8000 road-detection-backend
```

### Backend Troubleshooting

#### Common Issues

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

#### Monitoring

The backend provides monitoring endpoints:
- `/health` for system status
- Performance metrics
- Resource utilization

### Security

- Input validation
- Rate limiting
- CORS configuration
- Authentication support

## API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Object Detection
`POST /detect_objects/`

Detects objects in an uploaded image.

##### Request
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - file: Image file (JPG/PNG)

##### Response
```json
{
  "objects": [
    {
      "class": "car",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "caption": "A busy street with multiple vehicles",
  "processing_time": 0.45
}
```

#### Image Segmentation
`POST /segment_image/`

Performs semantic segmentation on an image.

##### Request
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - file: Image file (JPG/PNG)

##### Response
```json
{
  "segments": [
    {
      "class": "road",
      "area_percentage": 45.5
    }
  ],
  "mask_base64": "base64_encoded_mask_image"
}
```

#### Video Processing
`POST /process_video/`

Process video for object detection.

##### Request
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - file: Video file (MP4)

##### Response
```json
{
  "video_info": {
    "duration": 10.5,
    "fps": 30,
    "total_frames": 315,
    "processed_frames": 315
  },
  "detected_objects": {
    "car": 45,
    "person": 12
  },
  "summary_caption": "Video shows urban traffic with multiple vehicles and pedestrians"
}
```

#### Health Check
`GET /health`

Check API and model status.

##### Response
```json
{
  "status": "healthy",
  "models": ["fasterrcnn", "deeplabv3"]
}
```

### Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request - Invalid input |
| 415  | Unsupported Media Type |
| 500  | Internal Server Error |
| 503  | Service Unavailable - Model loading error |

### Rate Limiting

- 100 requests per minute per IP
- Burst: 25 requests
- Headers: X-RateLimit-Limit, X-RateLimit-Remaining

### Authentication

Currently uses CORS for access control. Token-based authentication planned for future releases.

## Developer Guide

### Development Environment Setup

1. **Required Tools**
   - VS Code or PyCharm
   - Git
   - Python 3.12+
   - Node.js (Latest LTS)
   - Docker (optional)

2. **Recommended VS Code Extensions**
   - Python
   - Pylance
   - ESLint
   - Prettier
   - GitLens
   - Docker

### Project Structure

```
├── app/                  # Next.js application files
├── backend/             # Python backend
├── components/          # React components
├── docs/                # Documentation
├── lib/                 # Shared utilities
└── tests/               # Test suites
```

### Development Workflow

#### Git Workflow

1. Create feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make changes and commit:
```bash
git add .
git commit -m "feat: your descriptive message"
```

3. Push and create PR:
```bash
git push origin feature/your-feature-name
```

#### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Tests
- chore: Maintenance

### Testing

#### Backend Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run specific test file
pytest tests/test_api_server.py
```

#### Frontend Testing
```bash
# Run Jest tests
npm test

# Run with coverage
npm test -- --coverage
```

### Code Style

#### Python
- Follow PEP 8
- Use type hints
- Document with docstrings
- Maximum line length: 88 characters

#### TypeScript/JavaScript
- Use ESLint configuration
- Prefer TypeScript
- Use functional components
- Document with JSDoc

### Adding New Features

#### Backend
1. Create new endpoint in `api_server.py`
2. Add corresponding model support if needed
3. Write tests
4. Update API documentation

#### Frontend
1. Create new component
2. Add to page layout
3. Connect to API
4. Add tests
5. Update documentation

### Model Development

#### Adding New Models
1. Create model class in `backend/model_factory.py`
2. Implement required interfaces
3. Add configuration options
4. Write tests
5. Update documentation

#### Model Optimization
- Profile inference time
- Monitor memory usage
- Implement batching
- Use quantization when possible

### Performance Guidelines

#### Backend
- Use async/await
- Implement caching
- Optimize database queries
- Profile endpoints

#### Frontend
- Lazy load components
- Optimize images
- Monitor bundle size
- Use performance profiler

### Security Best Practices

1. **Input Validation**
   - Validate file types
   - Check file sizes
   - Sanitize inputs

2. **API Security**
   - Use CORS
   - Rate limiting
   - Input validation
   - Authentication

3. **Data Protection**
   - Secure file handling
   - Clean up temporary files
   - Protect sensitive data

### Deployment

#### Backend Deployment
```bash
# Build Docker image
docker build -t road-detection-backend .

# Run container
docker run -p 8000:8000 road-detection-backend
```

#### Frontend Deployment
```bash
# Build production
npm run build

# Start production server
npm start
```

### Monitoring and Logging

#### Backend Logging
- Use structured logging
- Monitor API endpoints
- Track model performance
- Log error details

#### Frontend Monitoring
- Use error boundaries
- Track performance metrics
- Monitor user interactions
- Log client errors

### Contributing Guidelines

1. Fork the repository
2. Create feature branch
3. Follow code style
4. Write tests
5. Update documentation
6. Submit PR

### Support

For technical support:
- Open GitHub issue
- Contact: exploit0xffff@gmail.com