"""
PlantGuard AI - Prediction Module
Loads trained models and performs real inference on leaf images.
"""
import json
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

MODELS_DIR = Path("models")
IMAGE_SIZE = (128, 128)
MIN_CONFIDENCE_FOR_DECISION = 0.65
MIN_MODEL_AGREEMENT = 2

# Disease treatment database
TREATMENTS = {
    "Apple___Apple_scab": "Apply fungicide (captan or myclobutanil) in early spring. Remove fallen leaves. Plant resistant varieties.",
    "Apple___Black_rot": "Prune dead wood and mummified fruits. Apply fungicide during bloom. Remove infected fruit promptly.",
    "Apple___Cedar_apple_rust": "Apply fungicide in spring. Remove nearby cedar trees if possible. Plant resistant varieties.",
    "Apple___healthy": "No treatment needed. Continue regular care and monitoring.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply strobilurin fungicide. Rotate crops. Use resistant hybrids. Till crop residue.",
    "Corn_(maize)___Common_rust_": "Apply foliar fungicide if severe. Plant resistant hybrids. Monitor humidity levels.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply fungicide at first sign. Rotate crops annually. Use resistant varieties.",
    "Corn_(maize)___healthy": "No treatment needed. Continue regular care.",
    "Grape___Black_rot": "Apply mancozeb or myclobutanil fungicide. Remove mummified berries. Improve canopy airflow.",
    "Grape___Esca_(Black_Measles)": "No cure available. Prune infected wood. Apply wound protectant. Remove severely infected vines.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper-based fungicide. Remove infected leaves. Improve air circulation.",
    "Grape___healthy": "No treatment needed. Continue regular care.",
    "Potato___Early_blight": "Apply chlorothalonil fungicide. Practice crop rotation. Remove infected plant debris.",
    "Potato___Late_blight": "Apply copper-based fungicide immediately. Destroy infected plants. Improve drainage.",
    "Potato___healthy": "No treatment needed. Continue regular care.",
    "Tomato___Bacterial_spot": "Apply copper-based bactericide. Use disease-free seeds. Avoid overhead irrigation.",
    "Tomato___Early_blight": "Apply chlorothalonil or mancozeb. Mulch around plants. Practice 3-year crop rotation.",
    "Tomato___Late_blight": "Apply copper fungicide immediately. Remove all infected material. Do not compost.",
    "Tomato___Leaf_Mold": "Improve ventilation. Reduce humidity. Apply chlorothalonil fungicide.",
    "Tomato___Septoria_leaf_spot": "Apply fungicide. Remove lower infected leaves. Avoid overhead watering.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticide or neem oil. Increase humidity. Introduce predatory mites.",
    "Tomato___Target_Spot": "Apply fungicide. Remove infected leaves. Improve air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly vectors. Remove infected plants. Use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants. Disinfect tools. Use resistant varieties. Control aphid vectors.",
    "Tomato___healthy": "No treatment needed. Continue regular care.",
}


# ─── Custom layers (must match train_models.py) ─────────────────────────────

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)

    def call(self, x):
        x = self.projection(x)
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, -1, self.embed_dim])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return config


class PositionalEncoding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            "pos_embed", shape=(1, self.num_patches, self.embed_dim),
            initializer="random_normal",
        )

    def call(self, x):
        return x + self.pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return config


CUSTOM_OBJECTS = {
    "PatchEmbedding": PatchEmbedding,
    "PositionalEncoding": PositionalEncoding,
}


class PlantDiseasePredictor:
    """Loads all trained models and runs real inference."""

    def __init__(self):
        self.cnn_model = None
        self.feature_extractor = None
        self.rf_model = None
        self.svm_model = None
        self.xgb_model = None
        self.vit_model = None
        self.hybrid_model = None
        self.class_names = []
        self.metrics = {}
        self._load_models()

    def _load_models(self):
        """Load all trained models from disk."""
        class_file = MODELS_DIR / "class_names.json"
        if not class_file.exists():
            raise FileNotFoundError("class_names.json not found. Run train_models.py first.")
        with open(class_file) as f:
            self.class_names = json.load(f)

        metrics_file = MODELS_DIR / "model_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                self.metrics = json.load(f)

        cnn_path = MODELS_DIR / "cnn_final.keras"
        if cnn_path.exists():
            self.cnn_model = models.load_model(str(cnn_path))
            self.feature_extractor = models.Model(
                inputs=self.cnn_model.input,
                outputs=self.cnn_model.layers[-3].output,
            )
            print("✅ CNN loaded")

        vit_path = MODELS_DIR / "vit_final.keras"
        if vit_path.exists():
            self.vit_model = models.load_model(str(vit_path), custom_objects=CUSTOM_OBJECTS)
            print("✅ ViT loaded")

        hybrid_path = MODELS_DIR / "hybrid_cnn_vit_final.keras"
        if hybrid_path.exists():
            self.hybrid_model = models.load_model(str(hybrid_path), custom_objects=CUSTOM_OBJECTS)
            print("✅ Hybrid CNN-ViT loaded")

        rf_path = MODELS_DIR / "random_forest.joblib"
        if rf_path.exists():
            self.rf_model = joblib.load(str(rf_path))
            print("✅ Random Forest loaded")

        svm_path = MODELS_DIR / "svm.joblib"
        if svm_path.exists():
            self.svm_model = joblib.load(str(svm_path))
            print("✅ SVM loaded")

        xgb_path = MODELS_DIR / "xgboost.joblib"
        if xgb_path.exists():
            self.xgb_model = joblib.load(str(xgb_path))
            print("✅ XGBoost loaded")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize(IMAGE_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def _format_label(self, label: str) -> str:
        return label.replace("___", " - ").replace("_", " ")

    def _predict_keras_model(self, model, name, img_array):
        """Run prediction on a Keras model and return formatted result."""
        proba = model.predict(img_array, verbose=0)[0]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        result = {
            "disease": self.class_names[class_idx],
            "confidence": round(confidence, 4),
        }
        if name == "CNN (MobileNetV2)":
            result["all_probabilities"] = {
                self.class_names[i]: round(float(p), 4)
                for i, p in enumerate(proba) if p > 0.01
            }
        return result

    def _build_uncertain_response(self, predictions, best_model_name, best_prediction):
        return {
            "predicted_disease": "Unable to confidently identify the disease",
            "raw_class": "unknown",
            "confidence": best_prediction["confidence"],
            "severity": "low",
            "treatment": "Please upload a clear photo of a single leaf in good lighting with the affected area visible. Avoid blurry images, dark backgrounds, or non-leaf photos.",
            "best_model": best_model_name,
            "model_predictions": {
                name: {
                    "disease": self._format_label(pred["disease"]),
                    "confidence": pred["confidence"],
                }
                for name, pred in predictions.items()
            },
        }

    def predict(self, image: Image.Image) -> dict:
        """Run all models on the image and return a consensus prediction."""
        img_array = self.preprocess_image(image)
        predictions = {}

        if self.cnn_model:
            predictions["CNN (MobileNetV2)"] = self._predict_keras_model(
                self.cnn_model, "CNN (MobileNetV2)", img_array
            )

        if self.vit_model:
            predictions["ViT"] = self._predict_keras_model(
                self.vit_model, "ViT", img_array
            )

        if self.hybrid_model:
            predictions["CNN-ViT Hybrid"] = self._predict_keras_model(
                self.hybrid_model, "CNN-ViT Hybrid", img_array
            )

        if self.feature_extractor:
            features = self.feature_extractor.predict(img_array, verbose=0)

            if self.rf_model:
                rf_proba = self.rf_model.predict_proba(features)[0]
                rf_idx = int(np.argmax(rf_proba))
                predictions["Random Forest"] = {
                    "disease": self.class_names[rf_idx],
                    "confidence": round(float(rf_proba[rf_idx]), 4),
                }

            if self.svm_model:
                svm_proba = self.svm_model.predict_proba(features)[0]
                svm_idx = int(np.argmax(svm_proba))
                predictions["SVM (RBF)"] = {
                    "disease": self.class_names[svm_idx],
                    "confidence": round(float(svm_proba[svm_idx]), 4),
                }

            if self.xgb_model:
                xgb_proba = self.xgb_model.predict_proba(features)[0]
                xgb_idx = int(np.argmax(xgb_proba))
                predictions["XGBoost"] = {
                    "disease": self.class_names[xgb_idx],
                    "confidence": round(float(xgb_proba[xgb_idx]), 4),
                }

        if not predictions:
            raise RuntimeError("No trained models are available for inference.")

        best_model_name, best_prediction = max(
            predictions.items(), key=lambda item: item[1]["confidence"]
        )

        vote_counts = {}
        for pred in predictions.values():
            vote_counts[pred["disease"]] = vote_counts.get(pred["disease"], 0) + 1

        consensus_label = max(
            vote_counts,
            key=lambda label: (
                vote_counts[label],
                max(pred["confidence"] for pred in predictions.values() if pred["disease"] == label),
            ),
        )
        consensus_votes = vote_counts[consensus_label]
        consensus_predictions = [
            (name, pred) for name, pred in predictions.items() if pred["disease"] == consensus_label
        ]
        selected_model_name, selected_prediction = max(
            consensus_predictions,
            key=lambda item: item[1]["confidence"],
        )
        consensus_confidence = sum(pred["confidence"] for _, pred in consensus_predictions) / len(consensus_predictions)

        if (
            consensus_votes < MIN_MODEL_AGREEMENT
            or selected_prediction["confidence"] < MIN_CONFIDENCE_FOR_DECISION
        ):
            return self._build_uncertain_response(predictions, best_model_name, best_prediction)

        treatment = TREATMENTS.get(
            consensus_label,
            "Consult a local agricultural expert for specific treatment recommendations.",
        )

        if "healthy" in consensus_label.lower():
            severity = "low"
        elif consensus_confidence > 0.85:
            severity = "high"
        elif consensus_confidence > 0.65:
            severity = "medium"
        else:
            severity = "low"

        return {
            "predicted_disease": self._format_label(consensus_label),
            "raw_class": consensus_label,
            "confidence": round(float(consensus_confidence), 4),
            "severity": severity,
            "treatment": treatment,
            "best_model": selected_model_name,
            "model_predictions": {
                name: {
                    "disease": self._format_label(pred["disease"]),
                    "confidence": pred["confidence"],
                }
                for name, pred in predictions.items()
            },
        }
