"""Grad-CAM utilities for TinyMaskCNN (Lab7 XAI)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image
import matplotlib.pyplot as plt
from src.data import IMAGENET_MEAN, IMAGENET_STD, IDX_TO_CLASS, build_transforms


class GradCAM:
    """Gradient-weighted class activation mapping for the last conv activations."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Registers forward/backward hooks on the selected convolutional layer.
        Args:
            model (nn.Module): Classifier in eval mode.
            target_layer (nn.Module): Layer producing spatial feature maps.
        Returns:
            None
        """
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _gin, gout) -> None:
        self.gradients = gout[0].detach()

    def generate(self, images: torch.Tensor, class_idx: int | None = None) -> Tuple[torch.Tensor, int]:
        """Computes Grad-CAM heatmap for a single-image batch.
        Args:
            images (torch.Tensor): Tensor of shape (1, 3, H, W).
            class_idx (int | None): Target class; defaults to predicted class.
        Returns:
            Tuple[torch.Tensor, int]: Heatmap (H, W) in [0,1] and used class index.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = functional.relu(cam)
        cam = functional.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu(), class_idx


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized CHW tensor to an HWC uint8 RGB array.
    Args:
        tensor (torch.Tensor): Image tensor (3, H, W).
    Returns:
        np.ndarray: RGB image in [0, 255].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlays a heatmap onto an RGB image.
    Args:
        rgb (np.ndarray): HxWx3 uint8 image.
        cam (np.ndarray): HxW float heatmap.
        alpha (float): Blend factor for the heatmap.
    Returns:
        np.ndarray: Blended HxWx3 uint8 image.
    """
    cmap = plt.get_cmap("jet")
    heat = (cmap(cam)[..., :3] * 255).astype(np.uint8)
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


def explain_paths(
    model: nn.Module,
    paths: List[Path],
    device: torch.device,
    out_dir: Path,
) -> List[dict]:
    """Saves Grad-CAM overlays for a list of image paths.
    Args:
        model (nn.Module): Trained TinyMaskCNN.
        paths (List[Path]): Image files to explain.
        device (torch.device): Compute device.
        out_dir (Path): Directory for PNG overlays.
    Returns:
        List[dict]: Per-image prediction metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Last Conv2d before AdaptiveAvgPool is features[12]
    target = model.features[12]
    cam_engine = GradCAM(model, target)
    transform = build_transforms(train=False)
    results = []
    model.eval()
    for path in paths:
        with Image.open(path) as img:
            rgb_pil = img.convert("RGB")
        tensor = transform(rgb_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        heatmap, class_idx = cam_engine.generate(tensor)
        rgb = denormalize(tensor[0])
        overlay = overlay_cam(rgb, heatmap.numpy())
        out_path = out_dir / f"gradcam_{path.stem}_{IDX_TO_CLASS[class_idx]}.png"
        Image.fromarray(overlay).save(out_path)
        results.append({
            "path": str(path),
            "pred": IDX_TO_CLASS[class_idx],
            "prob_with_mask": float(probs[0]),
            "prob_without_mask": float(probs[1]),
            "overlay": str(out_path),
        })
    return results
