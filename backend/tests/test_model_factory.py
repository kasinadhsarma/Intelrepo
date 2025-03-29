import pytest
import torch
from backend.model_factory import ModelFactory

def test_create_model():
    """Test model creation for different model types"""
    
    # Test YOLO model creation
    yolo_model = ModelFactory.create_model('yolo')
    assert isinstance(yolo_model, torch.nn.Module)
    assert not yolo_model.training  # Should be in eval mode
    
    # Test FasterRCNN model creation
    fasterrcnn_model = ModelFactory.create_model('fasterrcnn')
    assert isinstance(fasterrcnn_model, torch.nn.Module)
    assert not fasterrcnn_model.training
    
    # Test invalid model type
    with pytest.raises(ValueError):
        ModelFactory.create_model('invalid_model')

def test_model_device_placement():
    """Test that models are correctly placed on available device"""
    model = ModelFactory.create_model('fasterrcnn')
    
    # Check if model parameters are on the correct device
    device = next(model.parameters()).device
    expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device == expected_device

@pytest.mark.parametrize("model_type", ['yolo', 'fasterrcnn'])
def test_model_inference_shape(model_type):
    """Test model inference produces correct output shape"""
    model = ModelFactory.create_model(model_type)
    
    # Create dummy input
    dummy_input = torch.randn(3, 224, 224).unsqueeze(0)
    dummy_input = dummy_input.to(next(model.parameters()).device)
    
    # Run inference
    with torch.no_grad():
        if model_type == 'yolo':
            output = model(dummy_input)
        else:
            output = model([dummy_input])
    
    # Check output structure
    if model_type == 'fasterrcnn':
        assert isinstance(output, list)
        assert all(isinstance(pred, dict) for pred in output)
        assert all('boxes' in pred for pred in output)
        assert all('labels' in pred for pred in output)
        assert all('scores' in pred for pred in output)

@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_processing(batch_size):
    """Test model handling of different batch sizes"""
    model = ModelFactory.create_model('fasterrcnn')
    
    # Create batch of images
    dummy_batch = torch.randn(batch_size, 3, 224, 224)
    dummy_batch = dummy_batch.to(next(model.parameters()).device)
    
    # Process batch
    with torch.no_grad():
        outputs = model([img for img in dummy_batch])
    
    assert len(outputs) == batch_size
    for output in outputs:
        assert isinstance(output, dict)
        assert 'boxes' in output
        assert 'labels' in output
        assert 'scores' in output