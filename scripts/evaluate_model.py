"""Entry point: evaluate best checkpoint on test split (Lab6)."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import make_loaders
from src.evaluate import evaluate_checkpoint
from src.model import TinyMaskCNN


def main() -> None:
    """Loads best.pt and writes test metrics + evaluation plots.
    Args:
        None
    Returns:
        None
    """
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(os.environ.get("MASK_DATA_ROOT", str(ROOT / "data" / "raw")))
    ckpt_path = ROOT / "artifacts" / "checkpoints" / "best.pt"
    hist_path = ROOT / "artifacts" / "checkpoints" / "history.json"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    history_payload = json.loads(hist_path.read_text(encoding="utf-8"))
    model = TinyMaskCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    _train, _val, test_loader, _split = make_loaders(data_root, batch_size=16, num_workers=2)
    metrics = evaluate_checkpoint(
        model,
        test_loader,
        device,
        history_payload["history"],
        ROOT / "artifacts" / "plots",
        split_name="test",
    )
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "roc_auc", "pr_auc")}, indent=2))


if __name__ == "__main__":
    main()
