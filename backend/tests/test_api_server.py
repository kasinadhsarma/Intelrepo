import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
import torch
from backend.api_server import app, setup_models, COCO_CLASSES

client = TestClient(app)

@pytest.fixture
def mock_image():
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_detect_objects_endpoint(mock_image):
    """Test object detection endpoint with a mock image"""
    files = {"file": ("test.png", mock_image, "image/png")}
    response = client.post("/detect_objects/", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert "detected_objects" in data
    assert "caption" in data
    assert "model_type" in data
    assert "device" in data
    
    # Check response structure
    for obj in data["detected_objects"]:
        assert "label" in obj
        assert "confidence" in obj
        assert "bbox" in obj
        assert isinstance(obj["confidence"], float)
        assert isinstance(obj["bbox"], list)
        assert len(obj["bbox"]) == 4
        assert obj["label"] in COCO_CLASSES

def test_model_setup():
    """Test model initialization"""
    detection_model, segmentation_model, device = setup_models()
    assert isinstance(detection_model, torch.nn.Module)
    assert isinstance(segmentation_model, torch.nn.Module)
    assert isinstance(device, torch.device)
    assert detection_model.training == False  # Model should be in eval mode
    assert segmentation_model.training == False  # Model should be in eval mode