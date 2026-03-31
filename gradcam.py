"""
Grad-CAM engine for OptiVis (EfficientNet-B0 architecture).
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import get_model, DEVICE

GRAD_CAM_ALPHA = 0.55


class _GradCAMHook:
    def __init__(self):
        self.activations = None
        self.gradients   = None
        self._handles    = []

    def register(self, layer: nn.Module):
        self._handles.append(layer.register_forward_hook(self._save_act))
        self._handles.append(layer.register_full_backward_hook(self._save_grad))

    def remove(self):
        for h in self._handles: h.remove()

    def _save_act(self, m, i, o): self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients = go[0].detach()


def _get_target_layer(model):
    """
    Find the best layer for Grad-CAM in EfficientNet-B0.
    In timm's EfficientNet, 'conv_head' is the final convolution.
    """
    if hasattr(model, "conv_head"):
        return model.conv_head
    
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            return m
    return None


@torch.no_grad()
def _prepare_input(img_rgb):
    from preprocessing import get_model_input
    x = get_model_input(img_rgb)
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    else:
        x = x.float()

    if x.ndim == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)
    elif x.ndim == 4:
        x = x.permute(0, 3, 1, 2)
    
    return x.to(DEVICE)


def make_gradcam_figure(
    img_rgb: np.ndarray,
    model_input: np.ndarray, # Optional
    class_idx: int,
    simulated: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    model = get_model()
    target_layer = _get_target_layer(model)
    
    if target_layer is None or simulated:
        h, w = img_rgb.shape[:2]
        return np.zeros((h, w)), np.zeros((h, w, 3), dtype=np.uint8), img_rgb

    hook = _GradCAMHook()
    hook.register(target_layer)

    try:
        x = _prepare_input(img_rgb)
        x.requires_grad_(True)
        
        model.zero_grad()
        logits = model(x)
        score  = logits[0, class_idx]
        score.backward()

        grads = hook.gradients
        acts  = hook.activations

        if grads is None or acts is None:
            cam = np.zeros(img_rgb.shape[:2], dtype=np.float32)
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam     = (weights * acts).sum(dim=1).squeeze(0)
            cam     = torch.relu(cam).cpu().numpy()
            
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        hook.remove()

    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    
    heatmap_img = (plt.get_cmap("jet")(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay_img = cv2.addWeighted(img_rgb, 1 - GRAD_CAM_ALPHA, heatmap_img, GRAD_CAM_ALPHA, 0)

    return cam_resized, heatmap_img, overlay_img
