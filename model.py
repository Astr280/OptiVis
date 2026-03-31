"""
Local Model Engine — Using user's trained EfficientNet-B0 weights.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import timm

# ───── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "efficientnet_b0"
WEIGHTS_FILE = "dr_efficientnet_weights.pt"
CLASS_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── Cache ───────────────────────────────────────────────────────────────────
_model_cache = None


def build_model():
    """
    Constructs the EfficientNet-B0 model architecture.
    """
    print(f"[INFO] Building {MODEL_NAME} architecture...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=5)
    model.to(DEVICE)
    return model


def get_model():
    """
    Loads and caches the model with trained weights.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model = build_model()

    if os.path.exists(WEIGHTS_FILE):
        print(f"[SUCCESS] Loading trained weights from {WEIGHTS_FILE}...")
        try:
            model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {e}. Starting fresh.")
    else:
        print(f"[WARNING] Weights file {WEIGHTS_FILE} not found. Running with ImageNet defaults.")

    model.eval()
    _model_cache = model
    return _model_cache


def predict(img_rgb: np.ndarray, model_input=None) -> tuple[np.ndarray, int, float, bool]:
    """
    Inference wrapper for the local trained model.
    """
    model = get_model()
    
    if model_input is None:
        from preprocessing import get_model_input
        model_input = get_model_input(img_rgb)
    
    if not isinstance(model_input, torch.Tensor):
        x = torch.from_numpy(model_input).float()
    else:
        x = model_input.float()

    # Dimension fix (handle 4D batch or 3D image)
    if x.ndim == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)
    elif x.ndim == 4:
        x = x.permute(0, 3, 1, 2)
    
    x = x.to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    
    return probs, class_idx, confidence, False


def _simulate_prediction(img_rgb):
    return np.zeros(5)
