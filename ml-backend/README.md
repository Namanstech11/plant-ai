# PlantGuard AI - ML Backend

Real machine learning pipeline for plant disease detection using 4 models:
- **CNN (MobileNetV2)** — Transfer learning for image classification
- **Random Forest** — Trained on CNN-extracted features
- **SVM (RBF)** — Trained on CNN-extracted features  
- **XGBoost** — Trained on CNN-extracted features

## Setup

```bash
cd ml-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

The pipeline uses the **PlantVillage** dataset (38 classes). It downloads automatically on first run.

To use a local dataset, place class folders in `data/PlantVillage/`:
```
data/PlantVillage/
├── Apple___Apple_scab/
├── Apple___healthy/
├── Tomato___Late_blight/
└── ...
```

## Train Models

```bash
python train_models.py
```

This will:
1. Download/load the PlantVillage dataset
2. Train CNN with MobileNetV2 transfer learning (+ fine-tuning)
3. Extract 128-dim features from CNN penultimate layer
4. Train Random Forest, SVM, XGBoost on extracted features
5. Evaluate all models (accuracy, precision, recall, F1, confusion matrix)
6. Save models to `models/` directory
7. Generate comparison charts

## Run API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Upload image → disease prediction |
| GET | `/models/metrics` | Training metrics for all models |
| GET | `/predictions/history` | Prediction history |
| GET | `/health` | API health check |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf_image.jpg"
```

### Example Response

```json
{
  "predicted_disease": "Tomato - Late blight",
  "confidence": 0.9423,
  "severity": "high",
  "treatment": "Apply copper fungicide immediately...",
  "best_model": "CNN (MobileNetV2)",
  "model_predictions": {
    "CNN (MobileNetV2)": {"disease": "Tomato - Late blight", "confidence": 0.9423},
    "Random Forest": {"disease": "Tomato - Late blight", "confidence": 0.8891},
    "SVM (RBF)": {"disease": "Tomato - Late blight", "confidence": 0.8654},
    "XGBoost": {"disease": "Tomato - Late blight", "confidence": 0.9102}
  }
}
```

## Frontend Connection

Set `VITE_API_URL=http://localhost:8000` in `.env` when running the React frontend.
