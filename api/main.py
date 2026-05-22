# Run with: uvicorn api.main:app --reload --port 8000
# Docs at:  http://localhost:8000/docs

import os
import sys
import io
import time
import base64
import numpy as np
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

# Append parent dir for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.model import load_adapter
from src.dataset import get_class_names
from src.gradcam import generate_gradcam, overlay_gradcam

# --- Lifespan for Model Loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    print("Loading model and classes...")
    config = Config()
    app.state.config = config
    
    app.state.class_names = get_class_names(config)
    
    # Load model
    if not os.path.exists(config.CHECKPOINT_DIR):
        print(f"Warning: Checkpoint directory {config.CHECKPOINT_DIR} does not exist.")
    app.state.model = load_adapter(config, config.CHECKPOINT_DIR)
    app.state.model.eval()
    
    print("Model loaded successfully.")
    yield
    # Cleanup
    print("Shutting down...")
    app.state.model = None

# --- App Setup ---
app = FastAPI(
    title="LoRA Image Classifier API",
    version="1.0.0",
    description="Plant disease classification with Grad-CAM explainability",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict[str, float]
    inference_time_ms: float

class ExplainResponse(BaseModel):
    predicted_class: str
    confidence: float
    gradcam_image_base64: str  # base64 encoded PNG
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    device: str

# --- Helper Functions ---
def _preprocess_image(file_bytes: bytes, image_size: int) -> torch.Tensor:
    """
    Preprocesses raw image bytes into a batch tensor [1, 3, H, W]
    """
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open image: {str(e)}")
        
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = val_transform(img).unsqueeze(0)
    return tensor

# --- Endpoints ---
@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """
    Returns API status, model loading status, number of classes, and device.
    """
    config = app.state.config
    is_loaded = hasattr(app.state, "model") and app.state.model is not None
    
    return HealthResponse(
        status="ok",
        model_loaded=is_loaded,
        num_classes=len(app.state.class_names),
        device=config.DEVICE
    )

@app.get("/classes", summary="Get Supported Classes")
async def get_classes():
    """
    Lists all supported plant disease classes.
    """
    return {
        "classes": app.state.class_names,
        "num_classes": len(app.state.class_names)
    }

@app.post("/predict", response_model=PredictionResponse, summary="Predict Plant Disease")
async def predict(image: UploadFile = File(...)):
    """
    Accepts an image upload, runs inference, and returns predicted class with probabilities.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    start_time = time.time()
    config = app.state.config
    
    try:
        file_bytes = await image.read()
        tensor = _preprocess_image(file_bytes, config.IMAGE_SIZE).to(config.DEVICE)
        
        with torch.no_grad():
            outputs = app.state.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
        conf_idx = int(np.argmax(probs))
        confidence = float(probs[conf_idx] * 100)
        pred_class = app.state.class_names[conf_idx]
        
        all_probs = {app.state.class_names[i]: float(probs[i] * 100) for i in range(len(probs))}
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predicted_class=pred_class,
            confidence=confidence,
            all_probabilities=all_probs,
            inference_time_ms=inference_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplainResponse, summary="Generate Grad-CAM Explanation")
async def explain(image: UploadFile = File(...)):
    """
    Accepts an image upload, runs inference, and returns a Grad-CAM overlay heatmap 
    encoded as base64.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    start_time = time.time()
    config = app.state.config
    
    try:
        file_bytes = await image.read()
        tensor = _preprocess_image(file_bytes, config.IMAGE_SIZE).to(config.DEVICE)
        
        with torch.no_grad():
            outputs = app.state.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            conf_idx = int(torch.argmax(probs).item())
            confidence = float(probs[conf_idx].item() * 100)
            pred_class = app.state.class_names[conf_idx]
            
        # Generate Grad-CAM
        cam = generate_gradcam(app.state.model, tensor, conf_idx, config.DEVICE)
        
        # Save temp file
        os.makedirs(config.GRADCAM_OUTPUT_DIR, exist_ok=True)
        temp_path = os.path.join(config.GRADCAM_OUTPUT_DIR, "temp_explain.png")
        
        # Save original image temp to create overlay (overlay_gradcam takes a path)
        temp_orig_path = os.path.join(config.GRADCAM_OUTPUT_DIR, "temp_orig.png")
        with open(temp_orig_path, "wb") as f:
            f.write(file_bytes)
            
        overlay_gradcam(temp_orig_path, cam, temp_path, pred_class, confidence)
        
        # Read back and encode
        with open(temp_path, "rb") as f:
            encoded_str = base64.b64encode(f.read()).decode('utf-8')
            
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_orig_path):
            os.remove(temp_orig_path)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return ExplainResponse(
            predicted_class=pred_class,
            confidence=confidence,
            gradcam_image_base64=encoded_str,
            inference_time_ms=inference_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
