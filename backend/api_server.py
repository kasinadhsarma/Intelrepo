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

# Initialize Ollama with Gemma3 model
llm = OllamaLLM(model="gemma3", temperature=0.7)

# Initialize models
def setup_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Object detection model
    detection_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    detection_model.eval()
    detection_model.to(device)
    
    # Segmentation model
    segmentation_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    segmentation_model.eval()
    segmentation_model.to(device)
    
    return detection_model, segmentation_model, device

detection_model, segmentation_model, device = setup_models()

# Caption generation chain using LangChain syntax
road_caption_prompt = PromptTemplate.from_template(
    "Based on object detection results in a road scene, describe what's happening in this traffic scenario. "
    "The scene contains: {objects}. Create a detailed caption focusing on the road environment, "
    "traffic conditions, potential hazards, and spatial relationships between vehicles, pedestrians, and infrastructure. "
    "Also mention any potential safety concerns or traffic rule violations if applicable. "
    "Do not ask for an image - just generate the caption based on the objects listed."
)
caption_chain = road_caption_prompt | llm | StrOutputParser()

# Helper function to convert image to base64
def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def calculate_safety_score(detected_objects):
    """
    Calculate a simple safety score based on detected objects
    Higher score means potentially safer scene (fewer objects, less crowded)
    """
    # Count objects by category
    vehicles = len([obj for obj in detected_objects if obj['label'] in ['car', 'truck', 'bus', 'motorcycle']])
    pedestrians = len([obj for obj in detected_objects if obj['label'] == 'person'])
    traffic_elements = len([obj for obj in detected_objects if obj['label'] in ['traffic light', 'stop sign']])
    
    # More objects generally means more complex scene with potentially more hazards
    total_objects = len(detected_objects)
    
    # Basic scoring logic (can be refined)
    if total_objects == 0:
        return 100  # Empty road, highest safety
    
    # Reduce score based on number of objects and their types
    base_score = 100
    base_score -= min(50, vehicles * 5)  # More vehicles reduce safety
    base_score -= min(30, pedestrians * 10)  # Pedestrians reduce safety more
    
    # Having traffic elements improves safety
    base_score += min(20, traffic_elements * 5)
    
    # Ensure score is between 0-100
    return max(0, min(100, base_score))

@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint for road object detection and caption generation
    Returns detected objects, their locations, and an AI-generated caption
    """
    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for model
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            predictions = detection_model([img_tensor])
        
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
        caption = caption_chain.invoke({"objects": objects_text})
        
        return {
            'detected_objects': detected_objects,
            'caption': caption.strip(),
            'model_type': 'fasterrcnn',
            'device': str(device),
            'object_categories': {
                'vehicles': len([obj for obj in detected_objects if obj['label'] in ['car', 'truck', 'bus', 'motorcycle']]),
                'pedestrians': len([obj for obj in detected_objects if obj['label'] == 'person']),
                'traffic_elements': len([obj for obj in detected_objects if obj['label'] in ['traffic light', 'stop sign']]),
                'cyclists': len([obj for obj in detected_objects if obj['label'] == 'bicycle']),
            },
            'safety_score': calculate_safety_score(detected_objects),  # You would need to implement this function
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment_image/")
async def segment_image(file: UploadFile = File(...)):
    """
    Endpoint for road scene segmentation
    Returns original image, segmented image, and mask image
    """
    try:
        start_time = time.time()
        
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for model
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(device)
        
        # Perform segmentation
        with torch.no_grad():
            output = segmentation_model(img_tensor)['out'][0]
        
        # Get segmentation mask
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Count unique segments (excluding background)
        unique_segments = np.unique(output_predictions)
        segments_count = len(unique_segments) - (1 if 0 in unique_segments else 0)
        
        # Create colored segmentation mask
        r = np.zeros_like(output_predictions).astype(np.uint8)
        g = np.zeros_like(output_predictions).astype(np.uint8)
        b = np.zeros_like(output_predictions).astype(np.uint8)
        
        # Road-specific color scheme
        color_map = {
            0: (0, 0, 0),      # background
            1: (128, 64, 128),  # road
            2: (244, 35, 232),  # sidewalk
            3: (70, 70, 70),    # building
            4: (102, 102, 156), # wall
            5: (190, 153, 153), # fence
            6: (153, 153, 153), # pole
            7: (250, 170, 30),  # traffic light
            8: (220, 220, 0),   # traffic sign
            9: (107, 142, 35),  # vegetation
            10: (152, 251, 152), # terrain
            11: (70, 130, 180),  # sky
            12: (220, 20, 60),   # person
            13: (255, 0, 0),     # rider
            14: (0, 0, 142),     # car
            15: (0, 0, 70),      # truck
            16: (0, 60, 100),    # bus
            17: (0, 80, 100),    # train
            18: (0, 0, 230),     # motorcycle
            19: (119, 11, 32),   # bicycle
        }
        
        # Assign colors to segments
        for segment_id in unique_segments:
            if segment_id in color_map:
                r_val, g_val, b_val = color_map[segment_id]
            else:
                r_val = np.random.randint(0, 255)
                g_val = np.random.randint(0, 255)
                b_val = np.random.randint(0, 255)
                
            r[output_predictions == segment_id] = r_val
            g[output_predictions == segment_id] = g_val
            b[output_predictions == segment_id] = b_val
        
        # Combine channels
        colored_mask = np.stack([r, g, b], axis=2)
        
        # Create segmented image (original with overlay)
        segmented_img = img_rgb.copy()
        mask = output_predictions > 0  # Non-background pixels
        segmented_img[mask] = cv2.addWeighted(segmented_img[mask], 0.5, colored_mask[mask], 0.5, 0)
        
        # Convert images to base64 for response
        original_base64 = image_to_base64(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        segmented_base64 = image_to_base64(cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
        mask_base64 = image_to_base64(colored_mask)
        
        processing_time = time.time() - start_time
        
        return {
            'original_image': original_base64,
            'segmented_image': segmented_base64,
            'mask_image': mask_base64,
            'segments_count': segments_count,
            'processing_time': processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    """
    Endpoint for processing video files
    Extracts frames, performs object detection, and returns analysis
    """
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
                img_tensor = img_tensor.to(device)
                
                # Perform detection
                with torch.no_grad():
                    predictions = detection_model([img_tensor])
                
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
            summary_caption = caption_chain.invoke({"objects": summary_objects})
            
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
    return {"status": "healthy", "models": ["fasterrcnn", "deeplabv3"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

