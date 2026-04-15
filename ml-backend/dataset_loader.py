"""
PlantVillage Dataset Loader
Downloads and prepares the PlantVillage dataset for training.
"""
import os
import shutil
import zipfile
import gdown
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_URL = "https://drive.google.com/uc?id=0B_voCy5O5sXMTFByemhpZllYREU"
DATASET_DIR = Path("data/PlantVillage")
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def download_dataset():
    """Download PlantVillage dataset if not present."""
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"Dataset already exists at {DATASET_DIR}")
        return

    print("Downloading PlantVillage dataset...")
    os.makedirs("data", exist_ok=True)
    zip_path = "data/plantvillage.zip"

    gdown.download(DATASET_URL, zip_path, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/")

    # Reorganize if needed - find the actual image directory
    for root, dirs, files in os.walk("data/"):
        if len(dirs) > 10:  # Found the class directories
            if root != str(DATASET_DIR):
                shutil.move(root, str(DATASET_DIR))
            break

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"Dataset ready at {DATASET_DIR}")


def get_class_names():
    """Return sorted list of class (disease) names from directory structure."""
    if not DATASET_DIR.exists():
        download_dataset()
    return sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])


def create_data_generators():
    """
    Create train/validation/test data generators with augmentation.
    Returns: (train_gen, val_gen, test_gen, class_names)
    """
    if not DATASET_DIR.exists():
        download_dataset()

    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    # No augmentation for validation/test
    test_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = test_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    class_names = list(train_gen.class_indices.keys())
    print(f"Found {len(class_names)} classes: {class_names[:5]}...")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    return train_gen, val_gen, class_names


def load_flat_dataset(max_per_class=500):
    """
    Load dataset as flat numpy arrays for sklearn models.
    Returns: (X_train, X_test, y_train, y_test, class_names)
    """
    from tensorflow.keras.utils import load_img, img_to_array

    if not DATASET_DIR.exists():
        download_dataset()

    images = []
    labels = []
    class_names = get_class_names()

    for idx, cls in enumerate(class_names):
        cls_dir = DATASET_DIR / cls
        files = list(cls_dir.glob("*.*"))[:max_per_class]
        for f in files:
            try:
                img = load_img(f, target_size=IMAGE_SIZE)
                arr = img_to_array(img) / 255.0
                images.append(arr)
                labels.append(idx)
            except Exception:
                continue

    X = np.array(images)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Flat dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test, class_names
