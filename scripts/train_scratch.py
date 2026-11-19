"""Entry point: train TinyMaskCNN from scratch on GPU if available."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import make_loaders
from src.train import train_model


def main() -> None:
    """Trains the from-scratch CNN and stores best checkpoint + history.
    Args:
        None
    Returns:
        None
    """
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    data_root = Path(os.environ.get("MASK_DATA_ROOT", str(ROOT / "data" / "raw")))
    print(f"data_root={data_root}")
    train_loader, val_loader, _test_loader, split = make_loaders(
        data_root,
        batch_size=16,
        num_workers=2,
        seed=42,
    )
    print(f"split train/val/test={len(split.train)}/{len(split.val)}/{len(split.test)}")
    result = train_model(
        train_loader,
        val_loader,
        device=device,
        epochs=10,
        lr=1e-3,
        patience=4,
        checkpoint_dir=ROOT / "artifacts" / "checkpoints",
    )
    print(result)


if __name__ == "__main__":
    main()
