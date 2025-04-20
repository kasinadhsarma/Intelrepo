"""
Optimized FastAPI server for Deep Actions Experimental with enhanced model handling and performance
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import cv2
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch
import torchvision.transforms.functional as F
import io
from PIL import Image
import base64
import time
import tempfile
import os
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

# COCO class names for object detection
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

ROAD_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter'
]

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP server
mcp_server = FastMCP()
app.include_router(mcp_server.router)

# Model Context Protocol Classes
class ModelMetadata(BaseModel):
    name: str
    version: str
    type: str
    framework: str
    date_initialized: str
    config: Dict[str, Any]

class ModelPerformanceMetrics(BaseModel):
    inference_count: int = 0
    avg_inference_time: float = 0.0
    total_inference_time: float = 0.0
    last_inference_time: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    detected_objects_count: Dict[str, int] = {}

class ModelContext:
    """Model Context Protocol for tracking model states and performances"""
    def __init__(self, model_name: str, model_type: str, model_version: str, framework: str, config: Dict[str, Any] = {}):
        self.metadata = ModelMetadata(
            name=model_name,
            version=model_version,
            type=model_type,
            framework=framework,
            date_initialized=datetime.now().isoformat(),
            config=config
        )
        self.performance = ModelPerformanceMetrics()
        self.is_ready = False
        self.context_history = []
        self.last_inputs = None
        self.last_outputs = None
    
    def record_inference(self, inference_time: float, detected_objects: List[Dict] = None):
        """Record inference statistics"""
        self.performance.inference_count += 1
        self.performance.total_inference_time += inference_time
        self.performance.avg_inference_time = self.performance.total_inference_time / self.performance.inference_count
        self.performance.last_inference_time = inference_time
        
        if detected_objects:
            for obj in detected_objects:
                label = obj.get('label')
                if label:
                    self.performance.detected_objects_count[label] = self.performance.detected_objects_count.get(label, 0) + 1
    
    def record_error(self, error_msg: str):
        """Record inference error"""
        self.performance.error_count += 1
        self.performance.last_error = error_msg
    
    def update_context(self, inputs: Any, outputs: Any):
        """Update context with recent inputs and outputs"""
        self.last_inputs = inputs
        self.last_outputs = outputs
        self.context_history.append({
            "timestamp": datetime.now().isoformat(),
            "input_type": str(type(inputs)),
            "output_type": str(type(outputs))
        })
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]
    
    def get_stats(self):
        """Get model statistics"""
        return {
            "metadata": self.metadata.dict(),
            "performance": self.performance.dict(),
            "is_ready": self.is_ready,
            "context_history_size": len(self.context_history)
        }
    
    def mark_ready(self):
        """Mark model as ready for inference"""
        self.is_ready = True

class ModelContextRegistry:
    """Registry to maintain all model contexts"""
    def __init__(self):
        self.contexts = {}
    
    def register_model(self, model_name: str, model_type: str, model_version: str, framework: str, config: Dict[str, Any] = {}):
        context = ModelContext(model_name, model_type, model_version, framework, config)
        self.contexts[model_name] = context
        return context
    
    def get_context(self, model_name: str):
        return self.contexts.get(model_name)
    
    def get_all_contexts(self):
        return {name: context.get_stats() for name, context in self.contexts.items()}

model_registry = ModelContextRegistry()

# Initialize models with warmup
def setup_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Detection model setup
    detection_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detection_model = fasterrcnn_resnet50_fpn(weights=detection_weights)
    detection_model.eval().to(device)
    
    # Segmentation model setup
    segmentation_weights = DeepLabV3_ResNet50_Weights.DEFAULT
    segmentation_model = deeplabv3_resnet50(weights=segmentation_weights)
    segmentation_model.eval().to(device)
    
    # Model warmup
    with torch.no_grad():
        dummy_input = [torch.rand(3, 224, 224, device=device)]
        detection_model(dummy_input)
        segmentation_model(torch.rand(1, 3, 224, 224, device=device))
    
    # Register models
    detection_context = model_registry.register_model(
        "fasterrcnn_resnet50", "object_detection", "resnet50_fpn", "pytorch",
        {"device": str(device), "confidence_threshold": 0.5}
    )
    segmentation_context = model_registry.register_model(
        "deeplabv3_resnet50", "segmentation", "resnet50", "pytorch",
        {"device": str(device)}
    )
    gemma_context = model_registry.register_model(
        "gemma3", "llm", "latest", "ollama", {"temperature": 0.7}
    )
    
    detection_context.mark_ready()
    segmentation_context.mark_ready()
    gemma_context.mark_ready()
    
    # Register models with MCP
    for model_name, context in model_registry.contexts.items():
        mcp_server.register_model(
            model_name=model_name,
            model_type=context.metadata.type,
            description=f"{context.metadata.name} {context.metadata.version} model",
            model=None,  # We're tracking models separately
            metadata={
                "framework": context.metadata.framework,
                "version": context.metadata.version,
                "config": context.metadata.config
            }
        )
    
    return detection_model, segmentation_model, detection_weights.transforms(), segmentation_weights.transforms(), device

detection_model, segmentation_model, detection_transforms, segmentation_transforms, device = setup_models()

# Initialize LLM
llm = OllamaLLM(model="gemma3", temperature=0.7)

# Caption generation chain
road_caption_prompt = PromptTemplate.from_template(
    "Based on object detection results in a road scene, describe what's happening in this traffic scenario. "
    "The scene contains: {objects}. Create a detailed caption focusing on the road environment, "
    "traffic conditions, potential hazards, and spatial relationships between vehicles, pedestrians, and infrastructure. "
    "Also mention any potential safety concerns or traffic rule violations if applicable."
)
caption_chain = road_caption_prompt | llm | StrOutputParser()

def image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def calculate_safety_score(detected_objects):
    vehicles = len([obj for obj in detected_objects if obj['label'] in ['car', 'truck', 'bus', 'motorcycle']])
    pedestrians = len([obj for obj in detected_objects if obj['label'] == 'person'])
    traffic_elements = len([obj for obj in detected_objects if obj['label'] in ['traffic light', 'stop sign']])
    
    base_score = 100
    base_score -= min(50, vehicles * 5)
    base_score -= min(30, pedestrians * 10)
    base_score += min(20, traffic_elements * 5)
    return max(0, min(100, base_score))

@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    start_time = time.time()
    detection_context = model_registry.get_context("fasterrcnn_resnet50")
    gemma_context = model_registry.get_context("gemma3")
    
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = img_pil.size
        
        # Apply transforms and get scaling factors
        img_transformed = detection_transforms(img_pil)
        img_tensor = img_transformed.unsqueeze(0).to(device)
        transformed_size = img_tensor.shape[-2:]
        scale_x = original_size[0] / transformed_size[1]
        scale_y = original_size[1] / transformed_size[0]
        
        # Log inference start with MCP
        inference_id = mcp_server.log_inference_start(
            model_name="fasterrcnn_resnet50",
            input_data={"image_width": original_size[0], "image_height": original_size[1]},
            metadata={"format": file.content_type}
        )
        
        with torch.no_grad():
            predictions = detection_model(img_tensor)
        
        detected_objects = []
        for pred in predictions:
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            for score, label, box in zip(scores, labels, boxes):
                if score > 0.5 and COCO_CLASSES[label] in ROAD_CLASSES:
                    scaled_box = [
                        box[0] * scale_x,
                        box[1] * scale_y,
                        box[2] * scale_x,
                        box[3] * scale_y
                    ]
                    detected_objects.append({
                        'label': COCO_CLASSES[label],
                        'confidence': float(score),
                        'bbox': [round(coord, 2) for coord in scaled_box]
                    })
        
        detection_time = time.time() - start_time
        detection_context.record_inference(detection_time, detected_objects)
        detection_context.update_context(
            {"image_size": original_size},
            {"detected_objects_count": len(detected_objects)}
        )
        
        # Log inference completion with MCP
        mcp_server.log_inference_end(
            inference_id=inference_id,
            output_data={"detected_objects": len(detected_objects)},
            metrics={"detection_time": detection_time},
            metadata={"objects": [obj['label'] for obj in detected_objects]}
        )
        
        # Generate caption asynchronously
        objects_text = ", ".join([obj['label'] for obj in detected_objects])
        caption_start = time.time()
        
        # Log LLM inference start with MCP
        llm_inference_id = mcp_server.log_inference_start(
            model_name="gemma3",
            input_data={"objects": objects_text},
            metadata={"type": "caption_generation"}
        )
        
        caption = await run_in_threadpool(caption_chain.invoke, {"objects": objects_text})
        caption_time = time.time() - caption_start
        
        gemma_context.record_inference(caption_time)
        gemma_context.update_context(
            {"objects": objects_text},
            {"caption_length": len(caption)}
        )
        
        # Log LLM inference end with MCP
        mcp_server.log_inference_end(
            inference_id=llm_inference_id,
            output_data={"caption_length": len(caption)},
            metrics={"caption_time": caption_time},
            metadata={"sentiment": "neutral"}  # Add actual sentiment analysis if available
        )
        
        return {
            'detected_objects': detected_objects,
            'caption': caption.strip(),
            'object_categories': {
                'vehicles': len([obj for obj in detected_objects if obj['label'] in ['car', 'truck', 'bus', 'motorcycle']]),
                'pedestrians': len([obj for obj in detected_objects if obj['label'] == 'person']),
                'traffic_elements': len([obj for obj in detected_objects if obj['label'] in ['traffic light', 'stop sign']]),
            },
            'safety_score': calculate_safety_score(detected_objects),
            'processing_metrics': {
                'detection_time': detection_time,
                'caption_time': caption_time,
                'total_time': time.time() - start_time
            }
        }
        
    except Exception as e:
        detection_context.record_error(str(e))
        gemma_context.record_error(str(e))
        # Log error in MCP
        mcp_server.log_error(
            model_name="fasterrcnn_resnet50",
            error_message=str(e),
            metadata={"endpoint": "/detect_objects/"}
        )
        raise HTTPException(500, str(e))

@app.post("/segment_image/")
async def segment_image(file: UploadFile = File(...)):
    start_time = time.time()
    segmentation_context = model_registry.get_context("deeplabv3_resnet50")
    
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = img_pil.size
        
        # Log inference start with MCP
        inference_id = mcp_server.log_inference_start(
            model_name="deeplabv3_resnet50",
            input_data={"image_width": original_size[0], "image_height": original_size[1]},
            metadata={"format": file.content_type}
        )
        
        # Apply transforms and process
        img_transformed = segmentation_transforms(img_pil)
        img_tensor = img_transformed.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = segmentation_model(img_tensor)['out']
        
        # Resize output to original dimensions
        output = torch.nn.functional.interpolate(
            output,
            size=original_size[::-1],
            mode='bilinear',
            align_corners=False
        )
        output_predictions = output.argmax(1).squeeze().cpu().numpy()
        
        # Vectorized mask generation
        color_map = {
            0: (0, 0, 0), 1: (128, 64, 128), 7: (250, 170, 30),
            8: (220, 220, 0), 12: (220, 20, 60), 14: (0, 0, 142),
            15: (0, 0, 70), 16: (0, 60, 100), 18: (0, 80, 100),
            19: (119, 11, 32)
        }
        color_lookup = np.zeros((256, 3), dtype=np.uint8)
        for idx, color in color_map.items():
            color_lookup[idx] = color
        colored_mask = color_lookup[output_predictions]
        
        # Create overlay
        original_img = np.array(img_pil)
        mask = output_predictions > 0
        segmented_img = cv2.addWeighted(original_img, 0.5, colored_mask, 0.5, 0)
        
        segmentation_time = time.time() - start_time
        segments_count = len(np.unique(output_predictions)) - 1
        
        # Log inference completion with MCP
        mcp_server.log_inference_end(
            inference_id=inference_id,
            output_data={"segments_count": segments_count},
            metrics={"segmentation_time": segmentation_time},
            metadata={"unique_classes": list(np.unique(output_predictions))}
        )
        
        # Encode images
        return {
            'original_image': image_to_base64(original_img),
            'segmented_image': image_to_base64(segmented_img),
            'mask_image': image_to_base64(colored_mask),
            'segments_count': segments_count,
            'processing_time': segmentation_time
        }
        
    except Exception as e:
        segmentation_context.record_error(str(e))
        # Log error in MCP
        mcp_server.log_error(
            model_name="deeplabv3_resnet50",
            error_message=str(e),
            metadata={"endpoint": "/segment_image/"}
        )
        raise HTTPException(500, str(e))

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    start_time = time.time()
    detection_context = model_registry.get_context("fasterrcnn_resnet50")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(await file.read())
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Log inference start with MCP
        inference_id = mcp_server.log_inference_start(
            model_name="fasterrcnn_resnet50",
            input_data={"video_frames": frame_count, "fps": fps},
            metadata={"format": file.content_type}
        )
        
        processed_frames = 0
        all_objects = []
        
        for frame_idx in range(0, frame_count, 30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            original_size = frame_pil.size
            img_transformed = detection_transforms(frame_pil)
            img_tensor = img_transformed.unsqueeze(0).to(device)
            
            # Bounding box scaling
            transformed_size = img_tensor.shape[-2:]
            scale_x = original_size[0] / transformed_size[1]
            scale_y = original_size[1] / transformed_size[0]
            
            with torch.no_grad():
                pred = detection_model(img_tensor)[0]
            
            # Process predictions with scaling
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                if score > 0.5 and COCO_CLASSES[label] in ROAD_CLASSES:
                    scaled_box = [
                        box[0].item() * scale_x,
                        box[1].item() * scale_y,
                        box[2].item() * scale_x,
                        box[3].item() * scale_y
                    ]
                    all_objects.append({
                        'frame': frame_idx,
                        'label': COCO_CLASSES[label],
                        'confidence': score.item(),
                        'bbox': [round(coord, 2) for coord in scaled_box]
                    })
            processed_frames += 1
        
        cap.release()
        os.unlink(video_path)
        
        processing_time = time.time() - start_time
        detection_context.record_inference(processing_time, all_objects)
        
        # Log inference completion with MCP
        mcp_server.log_inference_end(
            inference_id=inference_id,
            output_data={"processed_frames": processed_frames, "detected_objects": len(all_objects)},
            metrics={"processing_time": processing_time},
            metadata={"object_types": list(set(obj['label'] for obj in all_objects))}
        )
        
        return {
            'processed_frames': processed_frames,
            'detected_objects': all_objects,
            'object_counts': {obj['label']: sum(1 for o in all_objects if o['label'] == obj['label']) for obj in all_objects}
        }
        
    except Exception as e:
        detection_context.record_error(str(e))
        # Log error in MCP
        mcp_server.log_error(
            model_name="fasterrcnn_resnet50",
            error_message=str(e),
            metadata={"endpoint": "/process_video/"}
        )
        raise HTTPException(500, str(e))

# Model Context Protocol (MCP) additional endpoints
@app.get("/mcp/models/")
async def get_models():
    """Get all registered models with their contexts and MCP status"""
    mcp_models = mcp_server.list_models()
    context_models = model_registry.get_all_contexts()
    
    # Merge information
    for model_name in context_models:
        if model_name in mcp_models:
            context_models[model_name]["mcp_status"] = "registered"
        else:
            context_models[model_name]["mcp_status"] = "not_registered"
    
    return context_models

@app.get("/mcp/stats/")
async def get_mcp_stats():
    """Get MCP statistics including inference counts and performance metrics"""
    return {
        "mcp_version": mcp_server.version,
        "registered_models": len(mcp_server.list_models()),
        "inference_counts": {
            model_name: model_registry.get_context(model_name).performance.inference_count 
            for model_name in model_registry.contexts
        },
        "total_inferences": sum(
            model_registry.get_context(model_name).performance.inference_count 
            for model_name in model_registry.contexts
        ),
        "error_counts": {
            model_name: model_registry.get_context(model_name).performance.error_count 
            for model_name in model_registry.contexts
        }
    }

# Existing utility endpoints remain unchanged

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)