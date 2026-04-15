"""
PlantGuard AI - FastAPI Backend (Production Ready)
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
    description="ML-powered plant disease detection",
    version="2.0.0",
)

# ✅ CORS (important for Vercel frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Globals
predictor = None
prediction_history = []


# ✅ ROOT ROUTE (important for Railway)
@app.get("/")
async def root():
    return {"message": "🌱 PlantGuard AI backend is running"}


# ✅ HEALTH CHECK
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat(),
    }


# ✅ LOAD MODELS ON STARTUP
@app.on_event("startup")
async def load_models():
    global predictor
    try:
        print("🔄 Loading ML models...")

        predictor = PlantDiseasePredictor()

        print("✅ Models loaded successfully")

    except Exception as e:
        predictor = None
        print(f"❌ Model loading failed: {e}")


# ✅ PREDICTION ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check server logs.",
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # 🔥 Run ML prediction
        result = predictor.predict(image)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    # ✅ Log prediction
    log_entry = {
        "id": len(prediction_history) + 1,
        "timestamp": datetime.now().isoformat(),
        "image_name": file.filename,
        **result,
    }
    prediction_history.append(log_entry)

    return JSONResponse(content=result)


# ✅ MODEL METRICS
@app.get("/models/metrics")
async def get_model_metrics():
    metrics_file = Path("models/model_metrics.json")

    if not metrics_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Train models first.",
        )

    with open(metrics_file) as f:
        return json.load(f)


# ✅ HISTORY
@app.get("/predictions/history")
async def get_history():
    return {"predictions": prediction_history}


# ✅ LOCAL RUN (for dev only)
if __name__ == "__main__":
    import uvicorn

    import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)