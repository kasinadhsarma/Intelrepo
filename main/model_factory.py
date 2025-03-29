"""
Model Factory for Object Detection
================================

This module provides a factory class for creating and optimizing object detection models.
It demonstrates key concepts in deep learning model management:
1. Hardware detection and optimization
2. Model initialization and configuration
3. Memory management
4. Performance optimization techniques

The factory supports multiple model architectures (Faster R-CNN and YOLO) and
automatically configures them for optimal performance on available hardware.
"""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from yolo_detector import YOLODetector

class ModelFactory:
    """
    Factory class for creating optimized object detection models.

    This class demonstrates the Factory design pattern, which provides a clean interface
    for creating complex objects (in this case, deep learning models) while hiding the
    complexity of their initialization and configuration.
    """

    @staticmethod
    def create_model(model_type: str):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type.lower() == 'fasterrcnn':
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            model.to(device)
            return model, device
        elif model_type.lower() == 'yolo':
            return YOLODetector(), device
        else:
            raise ValueError(f"Unknown model type: {model_type}")