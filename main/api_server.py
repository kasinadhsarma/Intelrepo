"""
FastAPI server for Deep Actions Experimental with Langchain and Ollama integration
Provides API endpoints for object detection and caption generation using Gemma3 model
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch
import io
from PIL import Image
from model_factory import ModelFactory

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

# Initialize model with new weights parameter
def setup_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model, device

model, device = setup_model()

# Caption generation chain using new LangChain syntax
caption_prompt = PromptTemplate.from_template(
    "Based on object detection results, describe a scene containing: {objects}. Create a short, clear caption focusing on spatial relationships and activities. Do not ask for an image - just generate the caption based on the objects listed."
)
caption_chain = caption_prompt | llm | StrOutputParser()

@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint for object detection and caption generation
    Returns detected objects, their locations, and an AI-generated caption
    """
    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform object detection
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            predictions = model([img_tensor])
        
        # Process predictions
        predictions = [{k: v.to('cpu') for k, v in pred.items()} for pred in predictions]
        
        # Extract detected objects with high confidence
        detected_objects = []
        for pred in predictions:
            scores = pred['scores'].numpy()
            labels = pred['labels'].numpy()
            boxes = pred['boxes'].numpy()
            
            for score, label, box in zip(scores, labels, boxes):
                if score > 0.5:  # Confidence threshold
                    detected_objects.append({
                        'label': COCO_CLASSES[label],
                        'confidence': float(score),
                        'bbox': box.tolist()
                    })
        
        # Generate caption using Gemma3 model with new LangChain syntax
        objects_text = ", ".join([obj['label'] for obj in detected_objects])
        caption = caption_chain.invoke({"objects": objects_text})
        
        return {
            'detected_objects': detected_objects,
            'caption': caption.strip(),
            'model_type': 'fasterrcnn',
            'device': str(device)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)