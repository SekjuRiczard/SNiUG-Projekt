"""Exploratory data analysis for the face-mask dataset (Lab3)."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import discover_samples, stratified_split


def main() -> None:
    """Runs EDA: class counts, corrupt check, split sizes, sample grid.
    Args:
        None
    Returns:
        None
    """
    data_root = ROOT / "data" / "raw"
    plots = ROOT / "artifacts" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(data_root)
    counts = {k: len(v) for k, v in samples.items()}
    total = sum(counts.values())
    corrupt = []
    sizes = []
    for class_name, paths in samples.items():
        for path in paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    sizes.append(img.convert("RGB").size)
            except Exception as exc:  # noqa: BLE001
                corrupt.append({"path": str(path), "error": str(exc)})
    split = stratified_split(samples, seed=42)
    report = {
        "counts": counts,
        "total": total,
        "balance": {k: round(v / total, 4) for k, v in counts.items()},
        "corrupt_count": len(corrupt),
        "corrupt_examples": corrupt[:10],
        "split_sizes": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
        "split_class_train": dict(Counter(label for _, label in split.train)),
        "image_size_mode": Counter(sizes).most_common(5),
    }
    out_json = ROOT / "artifacts" / "reports" / "eda_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#2a9d8f", "#e76f51"])
    ax.set_title("Rozkład klas (dataset lokalny)")
    ax.set_ylabel("Liczba próbek")
    fig.tight_layout()
    fig.savefig(plots / "eda_class_balance.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for row, class_name in enumerate(samples):
        for col, path in enumerate(samples[class_name][:4]):
            with Image.open(path) as img:
                axes[row, col].imshow(img.convert("RGB"))
            axes[row, col].set_title(f"{class_name}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("Przykładowe próbki surowe")
    fig.tight_layout()
    fig.savefig(plots / "eda_samples.png", dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
