"""Evaluation metrics and plots for Lab6."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader
from src.data import IDX_TO_CLASS


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collects labels, predicted classes and softmax probabilities.
    Args:
        model (nn.Module): Trained classifier in eval mode.
        loader (DataLoader): Evaluation batches.
        device (torch.device): Compute device.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: y_true, y_pred, probs[:,1].
    """
    model.eval()
    ys: List[int] = []
    preds: List[int] = []
    probs_pos: List[float] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        ys.extend(labels.tolist())
        preds.extend(pred.cpu().tolist())
        probs_pos.extend(prob[:, 1].cpu().tolist())
    return np.asarray(ys), np.asarray(preds), np.asarray(probs_pos)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs_pos: np.ndarray) -> Dict:
    """Computes classification metrics for binary mask detection.
    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predicted labels.
        probs_pos (np.ndarray): P(without_mask) / class-1 probabilities.
    Returns:
        Dict: Scalar metrics and per-class precision/recall/f1.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, probs_pos)),
        "pr_auc": float(average_precision_score(y_true, probs_pos)),
        "per_class": {
            IDX_TO_CLASS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in (0, 1)
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=[IDX_TO_CLASS[0], IDX_TO_CLASS[1]], zero_division=0
        ),
    }


def save_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs_pos: np.ndarray,
    history: Dict,
    out_dir: Path,
) -> None:
    """Saves learning curves, confusion matrix, ROC and PR plots.
    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predictions.
        probs_pos (np.ndarray): Positive-class probabilities.
        history (Dict): Train/val loss history.
        out_dir (Path): Output directory for PNG files.
    Returns:
        None
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.get("train_loss", []), label="train")
    ax.plot(history.get("val_loss", []), label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=140)
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], [IDX_TO_CLASS[0], IDX_TO_CLASS[1]], rotation=20)
    ax.set_yticks([0, 1], [IDX_TO_CLASS[0], IDX_TO_CLASS[1]])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probs_pos)
    prec, rec, _ = precision_recall_curve(y_true, probs_pos)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr)
    axes[0].set_title("ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rec, prec)
    axes[1].set_title("Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr.png", dpi=140)
    plt.close(fig)


def evaluate_checkpoint(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    history: Dict,
    out_dir: Path,
    split_name: str = "test",
) -> Dict:
    """Runs full evaluation pipeline and writes metrics JSON + plots.
    Args:
        model (nn.Module): Loaded best model.
        loader (DataLoader): Split loader.
        device (torch.device): Device.
        history (Dict): Training history for curves.
        out_dir (Path): Artifacts directory.
        split_name (str): Name used in JSON filename.
    Returns:
        Dict: Computed metrics.
    """
    y_true, y_pred, probs = predict_loader(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, probs)
    save_evaluation_plots(y_true, y_pred, probs, history, out_dir)
    out_json = out_dir.parent / "reports" / f"metrics_{split_name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    serializable["classification_report"] = metrics["classification_report"]
    out_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(metrics["classification_report"])
    return metrics
