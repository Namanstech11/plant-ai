

# Plan: Add Vision Transformer (ViT) and CNN-ViT Hybrid Models

## Summary
Add two new models to the ML pipeline: a **Vision Transformer (ViT)** and a **CNN-ViT Hybrid** model. These will be trained alongside the existing 4 models, evaluated with the same metrics, and available for inference via the API.

## Changes

### 1. Update `ml-backend/requirements.txt`
- Add `vit-keras` or implement ViT manually using TensorFlow/Keras layers (MultiHeadAttention, patch embedding). Manual implementation is more reliable and avoids dependency issues.

### 2. Update `ml-backend/train_models.py`
Add two new model builders and training functions:

- **`build_vit(num_classes)`** — Vision Transformer:
  - Patch embedding layer (split 128x128 image into 8x8 or 16x16 patches)
  - Positional encoding
  - 4-6 Transformer encoder blocks (MultiHeadAttention + MLP + LayerNorm)
  - Global average pooling → Dense → Softmax
  
- **`build_hybrid_cnn_vit(num_classes)`** — CNN-ViT Hybrid:
  - Use MobileNetV2 as the CNN backbone to extract feature maps
  - Reshape CNN feature maps into patch-like tokens
  - Feed into 2-3 Transformer encoder blocks
  - Classification head (Dense → Softmax)

- Train both models with the same callbacks (EarlyStopping, ReduceLROnPlateau)
- Save as `vit_final.keras` and `hybrid_cnn_vit_final.keras`
- Evaluate both with accuracy/precision/recall/F1/confusion matrix
- Include them in the best-model selection

### 3. Update `ml-backend/predict.py`
- Load the two new models in `PlantDiseasePredictor.__init__`
- Add ViT and Hybrid inference in `predict()` method (same pattern as CNN — softmax output)
- Include their predictions in `model_predictions` response

### 4. Update `ml-backend/api.py`
- No structural changes needed — the predictor already returns all model predictions dynamically

### 5. Update Frontend (`src/components/ModelDashboard.tsx`)
- Extend the color palette array to support 6 models
- The dashboard already renders dynamically from the metrics array, so it will automatically show the new models

### 6. Update `src/pages/Index.tsx`
- No structural changes — it already maps over `model_predictions` dynamically

## Architecture Diagram

```text
Input Image (128x128)
    │
    ├──► CNN (MobileNetV2) ──► prediction
    │         │
    │         ├──► features ──► Random Forest ──► prediction
    │         ├──► features ──► SVM ──► prediction
    │         └──► features ──► XGBoost ──► prediction
    │
    ├──► ViT (Patch + Transformer) ──► prediction
    │
    └──► Hybrid (MobileNetV2 → Transformer) ──► prediction
```

## Files Modified
- `ml-backend/train_models.py` — add ViT and Hybrid build/train functions
- `ml-backend/predict.py` — load and run inference on 2 new models
- `ml-backend/requirements.txt` — no new deps needed (uses tf.keras built-in layers)
- `src/components/ModelDashboard.tsx` — extend color palette for 6 models

