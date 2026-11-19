"""CLI inference for a single image (Lab5 serving contract)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import IDX_TO_CLASS, build_transforms
from src.gradcam import GradCAM, denormalize, overlay_cam
from src.model import TinyMaskCNN


def main() -> None:
    """Runs Softmax inference (+ optional Grad-CAM) on one image path.
    Args:
        None
    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "artifacts" / "checkpoints" / "best.pt"))
    parser.add_argument("--gradcam", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMaskCNN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    path = Path(args.image)
    with Image.open(path) as img:
        rgb = img.convert("RGB")
    tensor = build_transforms(False)(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred = int(probs.argmax().item())
    result = {
        "image": str(path),
        "label": IDX_TO_CLASS[pred],
        "confidence_pct": round(float(probs[pred]) * 100, 2),
        "probs": {"with_mask": float(probs[0]), "without_mask": float(probs[1])},
    }
    if args.gradcam:
        cam_engine = GradCAM(model, model.features[12])
        heat, _ = cam_engine.generate(tensor, class_idx=pred)
        overlay = overlay_cam(denormalize(tensor[0]), heat.numpy())
        out = ROOT / "artifacts" / "plots" / f"infer_gradcam_{path.stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out)
        result["gradcam"] = str(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
