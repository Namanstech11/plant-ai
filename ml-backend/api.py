"""
PlantGuard AI - FastAPI Backend
Real ML inference API for plant disease detection.
"""
import io
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from predict import PlantDiseasePredictor

app = FastAPI(
    title="PlantGuard AI API",
    description="Real ML-powered plant disease detection from leaf images",
    version="2.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
predictor = None
prediction_history = []


@app.on_event("startup")
async def load_models():
    global predictor
    try:
        predictor = PlantDiseasePredictor()
        print("✅ All models loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠️ Models not found: {e}")
        print("Run 'python train_models.py' first to train models.")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a leaf image and get disease prediction from all 4 ML models.
    Returns real softmax/predict_proba confidence scores.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run train_models.py first.",
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run real ML inference
    result = predictor.predict(image)

    # Log prediction
    log_entry = {
        "id": len(prediction_history) + 1,
        "timestamp": datetime.now().isoformat(),
        "image_name": file.filename,
        **result,
    }
    prediction_history.append(log_entry)

    return JSONResponse(content=result)


@app.get("/models/metrics")
async def get_model_metrics():
    """Return real training metrics for all models."""
    metrics_file = Path("models/model_metrics.json")
    if not metrics_file.exists():
        raise HTTPException(
            status_code=404, detail="No metrics found. Train models first."
        )
    with open(metrics_file) as f:
        return json.load(f)


@app.get("/predictions/history")
async def get_history():
    """Return prediction history."""
    return {"predictions": prediction_history}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
