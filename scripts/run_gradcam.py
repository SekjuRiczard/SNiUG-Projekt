"""Generate Grad-CAM overlays for correct and incorrect predictions (Lab7)."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import IDX_TO_CLASS, make_loaders
from src.evaluate import predict_loader
from src.gradcam import explain_paths
from src.model import TinyMaskCNN


def main() -> None:
    """Explains at least three test images including one misclassification if any.
    Args:
        None
    Returns:
        None
    """
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(os.environ.get("MASK_DATA_ROOT", str(ROOT / "data" / "raw")))
    ckpt = torch.load(ROOT / "artifacts" / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = TinyMaskCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    _tr, _va, test_loader, split = make_loaders(data_root, batch_size=16, num_workers=2)
    y_true, y_pred, _probs = predict_loader(model, test_loader, device)
    paths = [p for p, _ in split.test]
    wrong = [paths[i] for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    correct = [paths[i] for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == yp]
    selected = (wrong[:1] + correct[:2] + wrong[1:2])[:4]
    if len(selected) < 3:
        selected = paths[:3]
    results = explain_paths(model, selected, device, ROOT / "artifacts" / "plots" / "gradcam")
    out = ROOT / "artifacts" / "reports" / "gradcam_examples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
