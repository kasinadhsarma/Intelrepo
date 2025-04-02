# Installation Guide

## Prerequisites

- Python 3.12+
- Node.js (Latest LTS version)
- CUDA-capable GPU (recommended)
- Git

## Backend Setup

1. Clone the repository:
```bash
git repo clone https://github.com/kasinadhsarma/Intelrepo
cd Intelrepo
```

2. Set up Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
cd backend
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## Frontend Setup

1. Install Node.js dependencies:
```bash
npm install
# or
pnpm install
```

2. Run the development server:
```bash
npm run dev
# or
pnpm dev
```

The application will be available at `http://localhost:3000`

## Model Setup

The system uses pre-trained models that will be downloaded automatically on first use:
- Faster R-CNN (ResNet50 backbone)
- YOLO (if configured)

## Configuration

### Backend Configuration
- Edit `backend/api_server.py` for API settings
- Modify model parameters in `backend/model_factory.py`

### Frontend Configuration
- Environment variables can be set in `.env.local`
- API endpoint configuration in `lib/api-client.ts`

## Troubleshooting

### Common Issues

1. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. Port Conflicts
```bash
# Check if ports are in use
netstat -ano | findstr "8000"  # Windows
lsof -i :8000                  # Linux/Mac
```

3. Model Download Issues
- Ensure internet connectivity
- Check write permissions in the model cache directory

### Getting Help

For additional help:
- Open an issue on GitHub
- Contact support at exploit0xffff@gmail.com