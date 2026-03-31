"""
Preprocessing utilities for retinal fundus images.
Applies CLAHE, green-channel enhancement, and normalization.
"""

import cv2
import numpy as np
from PIL import Image


TARGET_SIZE = (224, 224)


def load_image(uploaded_file) -> np.ndarray:
    """Load image from Streamlit UploadedFile and return as RGB numpy array."""
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the green channel to enhance retinal features."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def remove_black_border(img_rgb: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop black borders from fundus images."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_rgb = img_rgb[y : y + h, x : x + w]
    return img_rgb


def preprocess_image(img_rgb: np.ndarray, enhance: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Black border removal
    2. CLAHE enhancement
    3. Resize to 224x224
    4. Normalize to [0, 1]

    Returns: float32 numpy array of shape (224, 224, 3)
    """
    if enhance:
        img_rgb = remove_black_border(img_rgb)
        img_rgb = apply_clahe(img_rgb)

    img_resized = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized


def get_model_input(img_rgb: np.ndarray) -> np.ndarray:
    """Return batch-ready tensor: shape (1, 224, 224, 3)."""
    preprocessed = preprocess_image(img_rgb)
    return np.expand_dims(preprocessed, axis=0)
