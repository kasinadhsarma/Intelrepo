"""
FastAPI server for Deep Actions Experimental with Langchain and Ollama integration
Provides API endpoints for road object detection, caption generation, and image segmentation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch
import io
from PIL import Image
import base64
import time
import tempfile
import os

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

# Road-specific classes of interest
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

# Model management
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.detection_model = None
            self.segmentation_model = None
            self.llm = None
            self.caption_chain = None
            self.initialized = True
            self.status = "Not initialized"
            self.last_error = None
            try:
                self.setup_models()
                self.setup_llm()
                self.status = "Ready"
            except Exception as e:
                self.status = "Error"
                self.last_error = str(e)
                print(f"Error during initialization: {e}")

    def setup_models(self):
        try:
            # Object detection model
            self.detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.detection_model.eval()
            self.detection_model.to(self.device)
            
            # Segmentation model
            self.segmentation_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            self.segmentation_model.eval()
            self.segmentation_model.to(self.device)
        except Exception as e:
            self.last_error = str(e)
            print(f"Error initializing models: {str(e)}")
            raise

    def setup_llm(self):
        try:
            self.llm = OllamaLLM(model="gemma3", temperature=0.7)
            road_caption_prompt = PromptTemplate.from_template(
                "Based on object detection results in a road scene, describe what's happening in this traffic scenario. "
                "The scene contains: {objects}. Create a detailed caption focusing on the road environment, "
                "traffic conditions, potential hazards, and spatial relationships between vehicles, pedestrians, and infrastructure. "
                "Do not ask for an image - just generate the caption based on the objects listed."
            )
            self.caption_chain = road_caption_prompt | self.llm | StrOutputParser()
        except Exception as e:
            self.last_error = str(e)
            print(f"Error initializing LLM: {str(e)}")
            raise

    def cleanup(self):
        try:
            if self.detection_model:
                self.detection_model.to('cpu')
                del self.detection_model
            if self.segmentation_model:
                self.segmentation_model.to('cpu')
                del self.segmentation_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

# Helper function to convert image to base64
def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint for road object detection and caption generation
    Returns detected objects, their locations, and an AI-generated caption
    """
    if not model_manager.detection_model:
        raise HTTPException(status_code=500, detail="Detection model not initialized")

    if not model_manager.llm or not model_manager.caption_chain:
        raise HTTPException(status_code=500, detail="Language model not initialized")

    try:
        # Read and process image
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            # Prepare image for model
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(model_manager.device)
            
            with torch.no_grad():
                predictions = model_manager.detection_model([img_tensor])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")
        
        # Process predictions
        predictions = [{k: v.to('cpu') for k, v in pred.items()} for pred in predictions]
        
        # Extract detected objects with high confidence
        detected_objects = []
        for pred in predictions:
            scores = pred['scores'].numpy()
            labels = pred['labels'].numpy()
            boxes = pred['boxes'].numpy()
            
            for score, label, box in zip(scores, labels, boxes):
                if score > 0.5 and COCO_CLASSES[label] in ROAD_CLASSES:  # Confidence threshold and road-specific classes
                    detected_objects.append({
                        'label': COCO_CLASSES[label],
                        'confidence': float(score),
                        'bbox': box.tolist()
                    })
        
        # Generate caption using Gemma3 model with LangChain syntax
        objects_text = ", ".join([obj['label'] for obj in detected_objects])
        caption = model_manager.caption_chain.invoke({"objects": objects_text})
        
        return {
            'detected_objects': detected_objects,
            'caption': caption.strip(),
            'model_type': 'fasterrcnn',
            'device': str(model_manager.device)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    """
    Endpoint for processing video files
    Extracts frames, performs object detection, and returns analysis
    """
    if not model_manager.detection_model:
        raise HTTPException(status_code=500, detail="Detection model not initialized")

    if not model_manager.llm or not model_manager.caption_chain:
        raise HTTPException(status_code=500, detail="Language model not initialized")

    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            # Write the uploaded file content to the temporary file
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(temp_file_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Process a subset of frames (e.g., every 30th frame)
            frame_interval = 30
            frames_to_process = range(0, frame_count, frame_interval)
            
            # Initialize results
            all_objects = []
            processed_frames = 0
            
            for frame_idx in frames_to_process:
                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Prepare image for model
                img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.to(model_manager.device)
                
                # Perform detection
                with torch.no_grad():
                    predictions = model_manager.detection_model([img_tensor])
                
                # Process predictions
                predictions = [{k: v.to('cpu') for k, v in pred.items()} for pred in predictions]
                
                # Extract detected objects with high confidence
                frame_objects = []
                for pred in predictions:
                    scores = pred['scores'].numpy()
                    labels = pred['labels'].numpy()
                    boxes = pred['boxes'].numpy()
                    
                    for score, label, box in zip(scores, labels, boxes):
                        if score > 0.5 and COCO_CLASSES[label] in ROAD_CLASSES:
                            frame_objects.append({
                                'frame': frame_idx,
                                'timestamp': frame_idx / fps,
                                'label': COCO_CLASSES[label],
                                'confidence': float(score),
                                'bbox': box.tolist()
                            })
                
                all_objects.extend(frame_objects)
                processed_frames += 1
            
            # Close the video file
            cap.release()
            
            # Generate summary statistics
            object_counts = {}
            for obj in all_objects:
                label = obj['label']
                object_counts[label] = object_counts.get(label, 0) + 1
            
            # Generate a summary caption
            summary_objects = ", ".join([f"{count} {label}s" for label, count in object_counts.items()])
            summary_caption = model_manager.caption_chain.invoke({"objects": summary_objects})
            
            return {
                'video_info': {
                    'duration': duration,
                    'fps': fps,
                    'total_frames': frame_count,
                    'processed_frames': processed_frames
                },
                'detected_objects': all_objects,
                'object_counts': object_counts,
                'summary_caption': summary_caption.strip()
            }
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": model_manager.status,
        "device": str(model_manager.device),
        "models": {
            "detection": "loaded" if model_manager.detection_model else "not loaded",
            "segmentation": "loaded" if model_manager.segmentation_model else "not loaded",
            "llm": "loaded" if model_manager.llm else "not loaded"
        },
        "last_error": model_manager.last_error
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    model_manager.cleanup()

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        model_manager.cleanup()
