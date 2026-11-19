"""Training loop for TinyMaskCNN (Lab4)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from src.model import TinyMaskCNN, count_parameters


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    """Runs one train or eval epoch and returns mean loss.
    Args:
        model (nn.Module): Classifier.
        loader (DataLoader): Batch iterator.
        criterion (nn.Module): Loss module.
        device (torch.device): CUDA or CPU.
        optimizer (Optional[Optimizer]): If set, performs backward/step.
    Returns:
        float: Average loss over the epoch.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 12,
    lr: float = 1e-3,
    patience: int = 5,
    checkpoint_dir: Path | None = None,
) -> Dict:
    """Trains TinyMaskCNN with early stopping on validation loss.
    Args:
        train_loader (DataLoader): Training batches.
        val_loader (DataLoader): Validation batches.
        device (torch.device): Compute device.
        epochs (int): Max epochs.
        lr (float): AdamW learning rate.
        patience (int): Early-stopping patience.
        checkpoint_dir (Path | None): Where to store best.pt and history.
    Returns:
        Dict: History, best epoch/loss and parameter counts.
    """
    checkpoint_dir = checkpoint_dir or Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = TinyMaskCNN().to(device)
    total_p, train_p = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_epoch = -1
    stall = 0
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"epoch={epoch:02d} train={train_loss:.4f} val={val_loss:.4f} lr={history['lr'][-1]:.2e}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            stall = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "class_to_idx": {"with_mask": 0, "without_mask": 1},
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stall += 1
            if stall >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    history_path = checkpoint_dir / "history.json"
    payload = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "total_params": total_p,
        "trainable_params": train_p,
    }
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
