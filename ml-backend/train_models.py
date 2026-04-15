"""
PlantGuard AI - Full ML Training Pipeline
Trains CNN + Random Forest + SVM + XGBoost + ViT + CNN-ViT Hybrid
on the PlantVillage dataset.
"""
import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataset_loader import create_data_generators, load_flat_dataset, IMAGE_SIZE

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ─── 1. CNN Model ────────────────────────────────────────────────────────────


def build_cnn(num_classes: int):
    """Build a CNN using MobileNetV2 transfer learning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(train_gen, val_gen, num_classes: int, epochs=20):
    """Train the CNN model."""
    print("\n" + "=" * 60)
    print("TRAINING CNN (MobileNetV2 Transfer Learning)")
    print("=" * 60)

    model = build_cnn(num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(
            str(MODELS_DIR / "cnn_best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Fine-tune: unfreeze top layers of base model
    print("\nFine-tuning base model layers...")
    model.layers[0].trainable = True
    for layer in model.layers[0].layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks,
    )

    model.save(str(MODELS_DIR / "cnn_final.keras"))
    print(f"CNN saved to {MODELS_DIR / 'cnn_final.keras'}")

    return model, history


# ─── 2. Feature Extraction ──────────────────────────────────────────────────


def build_feature_extractor(cnn_model):
    """Extract the feature extractor (everything before the final Dense)."""
    feature_model = models.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.layers[-3].output,
    )
    return feature_model


def extract_features(feature_model, X):
    """Extract features from image arrays using the CNN feature extractor."""
    print(f"Extracting features from {len(X)} images...")
    features = feature_model.predict(X, batch_size=32, verbose=1)
    print(f"Feature shape: {features.shape}")
    return features


# ─── 3. Traditional ML Models ───────────────────────────────────────────────


def train_random_forest(X_train, y_train):
    """Train Random Forest on CNN-extracted features."""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_split=5,
        n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, str(MODELS_DIR / "random_forest.joblib"))
    print("Random Forest saved.")
    return rf


def train_svm(X_train, y_train):
    """Train SVM on CNN-extracted features."""
    print("\n" + "=" * 60)
    print("TRAINING SVM (RBF Kernel)")
    print("=" * 60)

    svm = SVC(
        kernel="rbf", C=10.0, gamma="scale",
        probability=True, random_state=42,
    )
    svm.fit(X_train, y_train)
    joblib.dump(svm, str(MODELS_DIR / "svm.joblib"))
    print("SVM saved.")
    return svm


def train_xgboost(X_train, y_train, num_classes: int):
    """Train XGBoost on CNN-extracted features."""
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        objective="multi:softprob", num_class=num_classes,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, str(MODELS_DIR / "xgboost.joblib"))
    print("XGBoost saved.")
    return xgb


# ─── 4. Vision Transformer (ViT) ────────────────────────────────────────────


class PatchEmbedding(layers.Layer):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size,
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, x):
        x = self.projection(x)  # (B, H/P, W/P, embed_dim)
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, -1, self.embed_dim])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return config


class PositionalEncoding(layers.Layer):
    """Learnable positional encoding for transformer tokens."""

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


def transformer_encoder_block(x, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
    """Single Transformer encoder block."""
    # Multi-head self-attention
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate,
    )(x, x)
    attn_out = layers.Dropout(dropout_rate)(attn_out)
    x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    # MLP (feed-forward)
    mlp_out = layers.Dense(mlp_dim, activation="gelu")(x1)
    mlp_out = layers.Dropout(dropout_rate)(mlp_out)
    mlp_out = layers.Dense(embed_dim)(mlp_out)
    mlp_out = layers.Dropout(dropout_rate)(mlp_out)
    x2 = layers.LayerNormalization(epsilon=1e-6)(x1 + mlp_out)

    return x2


def build_vit(num_classes: int):
    """
    Build a Vision Transformer from scratch.
    Image 128x128 → 16x16 patches (64 patches) → 6 Transformer blocks → classification.
    """
    patch_size = 16
    embed_dim = 256
    num_heads = 8
    mlp_dim = 512
    num_layers = 6
    num_patches = (IMAGE_SIZE[0] // patch_size) * (IMAGE_SIZE[1] // patch_size)  # 64

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

    # Patch embedding
    x = PatchEmbedding(patch_size, embed_dim)(inputs)
    x = PositionalEncoding(num_patches, embed_dim)(x)
    x = layers.Dropout(0.1)(x)

    # Transformer encoder blocks
    for _ in range(num_layers):
        x = transformer_encoder_block(x, embed_dim, num_heads, mlp_dim, dropout_rate=0.1)

    # Classification head — global average pooling over patch tokens
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ViT")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_vit(train_gen, val_gen, num_classes: int, epochs=25):
    """Train the Vision Transformer."""
    print("\n" + "=" * 60)
    print("TRAINING VISION TRANSFORMER (ViT)")
    print("=" * 60)

    model = build_vit(num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(
            str(MODELS_DIR / "vit_best.keras"),
            save_best_only=True, monitor="val_accuracy",
        ),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
    ]

    model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs, callbacks=callbacks,
    )

    model.save(str(MODELS_DIR / "vit_final.keras"))
    print(f"ViT saved to {MODELS_DIR / 'vit_final.keras'}")
    return model


# ─── 5. CNN-ViT Hybrid Model ────────────────────────────────────────────────


def build_hybrid_cnn_vit(num_classes: int):
    """
    CNN-ViT Hybrid: MobileNetV2 extracts spatial features,
    then Transformer blocks refine the token representation.
    Feature maps from CNN are reshaped into patch-like tokens.
    """
    embed_dim = 128
    num_heads = 4
    mlp_dim = 256
    num_transformer_layers = 3

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

    # CNN backbone — MobileNetV2 feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze initially

    # Get CNN feature maps: (B, 4, 4, 1280) for 128x128 input
    cnn_features = base_model(inputs)

    # Reshape feature maps into tokens: (B, 16, 1280) → project to embed_dim
    B = tf.shape(cnn_features)[0]
    H, W, C = cnn_features.shape[1], cnn_features.shape[2], cnn_features.shape[3]
    num_tokens = H * W  # 4*4 = 16 tokens

    x = tf.reshape(cnn_features, [B, -1, C])  # (B, 16, 1280)
    x = layers.Dense(embed_dim)(x)  # Project to embed_dim: (B, 16, 128)
    x = PositionalEncoding(num_tokens, embed_dim)(x)
    x = layers.Dropout(0.1)(x)

    # Transformer encoder blocks
    for _ in range(num_transformer_layers):
        x = transformer_encoder_block(x, embed_dim, num_heads, mlp_dim, dropout_rate=0.1)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="Hybrid_CNN_ViT")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_hybrid(train_gen, val_gen, num_classes: int, epochs=20):
    """Train the CNN-ViT Hybrid model."""
    print("\n" + "=" * 60)
    print("TRAINING CNN-ViT HYBRID (MobileNetV2 + Transformer)")
    print("=" * 60)

    model = build_hybrid_cnn_vit(num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(
            str(MODELS_DIR / "hybrid_best.keras"),
            save_best_only=True, monitor="val_accuracy",
        ),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
    ]

    model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs, callbacks=callbacks,
    )

    # Fine-tune CNN backbone
    print("\nFine-tuning CNN backbone in hybrid model...")
    base_model_layer = model.layers[1]  # MobileNetV2
    base_model_layer.trainable = True
    for layer in base_model_layer.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen, validation_data=val_gen,
        epochs=10, callbacks=callbacks,
    )

    model.save(str(MODELS_DIR / "hybrid_cnn_vit_final.keras"))
    print(f"Hybrid saved to {MODELS_DIR / 'hybrid_cnn_vit_final.keras'}")
    return model


# ─── 6. Evaluation ──────────────────────────────────────────────────────────


def evaluate_model(name, model, X_test, y_test, class_names):
    """Evaluate a model and return metrics dict."""
    if name in ("CNN", "ViT", "Hybrid CNN-ViT"):
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }

    print(f"\n{'─' * 40}")
    print(f"Model: {name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_').replace('-', '_')}.png"))
    plt.close()

    return metrics


def select_best_model(all_metrics):
    """Select the best model based on F1 score."""
    best = max(all_metrics, key=lambda m: m["f1_score"])
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best['name']} (F1: {best['f1_score']:.4f})")
    print(f"{'=' * 60}")
    return best


# ─── 7. Main Training Pipeline ──────────────────────────────────────────────


def main():
    print("PlantGuard AI - Full ML Training Pipeline (6 Models)")
    print("=" * 60)

    # Step 1: Load dataset
    train_gen, val_gen, class_names = create_data_generators()
    num_classes = len(class_names)

    # Save class names
    with open(str(MODELS_DIR / "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # Step 2: Train CNN
    cnn_model, history = train_cnn(train_gen, val_gen, num_classes, epochs=20)

    # Step 3: Train Vision Transformer
    vit_model = train_vit(train_gen, val_gen, num_classes, epochs=25)

    # Step 4: Train CNN-ViT Hybrid
    hybrid_model = train_hybrid(train_gen, val_gen, num_classes, epochs=20)

    # Step 5: Load flat dataset for feature extraction
    X_train, X_test, y_train, y_test, _ = load_flat_dataset(max_per_class=500)

    # Re-encode labels to be contiguous (fixes XGBoost "Invalid classes" error)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    actual_num_classes = len(le.classes_)
    print(f"Re-encoded labels: {len(le.classes_)} contiguous classes")

    # Step 6: Extract features using CNN
    feature_extractor = build_feature_extractor(cnn_model)
    X_train_features = extract_features(feature_extractor, X_train)
    X_test_features = extract_features(feature_extractor, X_test)

    # Step 7: Train additional models on extracted features
    rf_model = train_random_forest(X_train_features, y_train)
    svm_model = train_svm(X_train_features, y_train)
    xgb_model = train_xgboost(X_train_features, y_train, actual_num_classes)

    # Step 8: Evaluate all 6 models
    all_metrics = []

    cnn_metrics = evaluate_model("CNN", cnn_model, X_test, y_test, class_names)
    all_metrics.append(cnn_metrics)

    vit_metrics = evaluate_model("ViT", vit_model, X_test, y_test, class_names)
    all_metrics.append(vit_metrics)

    hybrid_metrics = evaluate_model("Hybrid CNN-ViT", hybrid_model, X_test, y_test, class_names)
    all_metrics.append(hybrid_metrics)

    rf_metrics = evaluate_model("Random Forest", rf_model, X_test_features, y_test, class_names)
    all_metrics.append(rf_metrics)

    svm_metrics = evaluate_model("SVM", svm_model, X_test_features, y_test, class_names)
    all_metrics.append(svm_metrics)

    xgb_metrics = evaluate_model("XGBoost", xgb_model, X_test_features, y_test, class_names)
    all_metrics.append(xgb_metrics)

    # Step 9: Select best model
    best = select_best_model(all_metrics)

    # Save all metrics
    with open(str(MODELS_DIR / "model_metrics.json"), "w") as f:
        json.dump({"models": all_metrics, "best_model": best["name"]}, f, indent=2)

    # Save comparison chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    model_names = [m["name"] for m in all_metrics]
    bar_colors = ["#2d6a4f", "#e63946", "#457b9d", "#52b788", "#74c69d", "#95d5b2"]

    for ax, metric in zip(axes, metric_names):
        values = [m[metric] for m in all_metrics]
        bars = ax.bar(model_names, values, color=bar_colors)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / "model_comparison.png"), dpi=150)
    plt.close()

    print("\n✅ Training complete! All 6 models saved to ./models/")
    print("Run the API: cd ml-backend && uvicorn api:app --reload")


if __name__ == "__main__":
    main()
