import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import torch
import numpy as np
from PIL import Image
import io

@pytest.fixture(scope="session")
def device():
    """Provides the compute device for all tests"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture(scope="session")
def dummy_image():
    """Creates a dummy image for testing"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[50:150, 50:150] = 255  # Add a white square for detection
    return img

@pytest.fixture(scope="session")
def image_bytes(dummy_image):
    """Converts dummy image to bytes for API testing"""
    img = Image.fromarray(dummy_image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture(scope="session")
def torch_image(dummy_image):
    """Converts dummy image to torch tensor"""
    return torch.from_numpy(dummy_image).permute(2, 0, 1).float() / 255.0

def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as a slow test"
    )