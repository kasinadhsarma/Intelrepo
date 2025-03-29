import pytest
import cv2
import numpy as np
import torch
import threading
from unittest.mock import MagicMock, patch
from backend.main import (
    object_detection_setup,
    perform_object_detection,
    FrameCaptureThread,
    UpdateFrameThread,
    detect_objects_in_image
)

@pytest.fixture
def mock_frame():
    """Create a mock video frame"""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_model():
    """Create a mock object detection model"""
    model = MagicMock()
    model.eval = MagicMock(return_value=None)
    model.to = MagicMock(return_value=model)
    return model

def test_object_detection_setup():
    """Test model initialization with both model types"""
    # Test FasterRCNN
    model, device = object_detection_setup('fasterrcnn')
    assert isinstance(model, torch.nn.Module)
    assert isinstance(device, torch.device)
    assert not model.training

    # Test YOLO
    model, device = object_detection_setup('yolo')
    assert isinstance(model, torch.nn.Module)
    assert isinstance(device, torch.device)
    assert not model.training

def test_perform_object_detection(mock_frame, mock_model):
    """Test object detection pipeline"""
    device = torch.device('cpu')
    
    # Mock model predictions
    mock_predictions = [{
        'boxes': torch.tensor([[100, 100, 200, 200]]),
        'labels': torch.tensor([1]),
        'scores': torch.tensor([0.9])
    }]
    mock_model.return_value = mock_predictions
    
    # Test FasterRCNN detection
    frame_with_objects, predictions = perform_object_detection(
        mock_model, 
        device, 
        mock_frame, 
        'fasterrcnn'
    )
    
    assert isinstance(frame_with_objects, np.ndarray)
    assert isinstance(predictions, list)
    assert frame_with_objects.shape == mock_frame.shape

def test_frame_capture_thread():
    """Test frame capture threading"""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3)))
    
    capture_thread = FrameCaptureThread(mock_cap)
    capture_thread.start()
    
    # Let the thread run briefly
    import time
    time.sleep(0.1)
    
    # Check that frame is being captured
    assert capture_thread.frame is not None
    assert isinstance(capture_thread.frame, np.ndarray)
    
    # Test thread cleanup
    capture_thread.stop()
    capture_thread.join()
    assert not capture_thread.running
    mock_cap.release.assert_called_once()

def test_update_frame_thread():
    """Test GUI update thread"""
    mock_window = MagicMock()
    update_thread = UpdateFrameThread(mock_window)
    update_thread.start()
    
    # Let the thread run briefly
    import time
    time.sleep(0.1)
    
    # Check that frame updates are being called
    assert mock_window.update_frame.called
    
    # Test thread cleanup
    update_thread.stop()
    update_thread.join()
    assert not update_thread.running

@pytest.mark.integration
def test_detect_objects_in_image(tmp_path):
    """Integration test for image detection pipeline"""
    # Create a test image
    input_path = tmp_path / "test_input.jpg"
    output_path = tmp_path / "test_output.jpg"
    
    # Create and save a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.imwrite(str(input_path), test_image)
    
    # Run detection
    detect_objects_in_image(str(input_path), str(output_path), 'fasterrcnn')
    
    # Verify output was created
    assert output_path.exists()
    output_image = cv2.imread(str(output_path))
    assert output_image is not None
    assert output_image.shape == test_image.shape