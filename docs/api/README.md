# API Reference

## Base URL
```
http://localhost:8000
```

## Endpoints

### Object Detection
`POST /detect_objects/`

Detects objects in an uploaded image.

#### Request
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - file: Image file (JPG/PNG)

#### Response
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

### Image Segmentation
`POST /segment_image/`

Performs semantic segmentation on an image.

#### Request
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - file: Image file (JPG/PNG)

#### Response
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

### Video Processing
`POST /process_video/`

Process video for object detection.

#### Request
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - file: Video file (MP4)

#### Response
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

### Health Check
`GET /health`

Check API and model status.

#### Response
```json
{
  "status": "healthy",
  "models": ["fasterrcnn", "deeplabv3"]
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request - Invalid input |
| 415  | Unsupported Media Type |
| 500  | Internal Server Error |
| 503  | Service Unavailable - Model loading error |

## Rate Limiting

- 100 requests per minute per IP
- Burst: 25 requests
- Headers: X-RateLimit-Limit, X-RateLimit-Remaining

## Authentication

Currently uses CORS for access control. Token-based authentication planned for future releases.