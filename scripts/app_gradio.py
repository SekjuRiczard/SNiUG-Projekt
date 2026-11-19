"""Gradio GUI: upload a face image and inspect TinyMaskCNN predictions + project plots."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IDX_TO_CLASS, build_transforms
from src.gradcam import GradCAM, denormalize, overlay_cam
from src.model import TinyMaskCNN

CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best.pt"
HISTORY = ROOT / "artifacts" / "reports" / "train_history.json"
METRICS = ROOT / "artifacts" / "reports" / "metrics_test.json"
PLOTS_DIR = ROOT / "artifacts" / "plots"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: Optional[TinyMaskCNN] = None
CAM_ENGINE: Optional[GradCAM] = None
TRANSFORM = build_transforms(train=False)


def load_model() -> TinyMaskCNN:
    """Loads the best checkpoint once and caches Grad-CAM hooks.
    Args:
        None
    Returns:
        TinyMaskCNN: Model in eval mode on DEVICE.
    """
    global MODEL, CAM_ENGINE
    if MODEL is not None:
        return MODEL
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Brak checkpointu: {CHECKPOINT}")
    model = TinyMaskCNN().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    MODEL = model
    CAM_ENGINE = GradCAM(model, model.features[12])
    return model


def probability_figure(probs: Dict[str, float]) -> plt.Figure:
    """Builds a bar chart of Softmax class probabilities.
    Args:
        probs (Dict[str, float]): Mapping class name -> probability.
    Returns:
        plt.Figure: Matplotlib figure with horizontal bars.
    """
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    names = list(probs.keys())
    values = [probs[n] for n in names]
    colors = ["#2a9d8f", "#e76f51"]
    ax.barh(names, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prawdopodobieństwo Softmax")
    ax.set_title("Rozkład klas")
    for y_pos, value in enumerate(values):
        ax.text(min(value + 0.02, 0.98), y_pos, f"{value * 100:.1f}%", va="center")
    fig.tight_layout()
    return fig


def load_metrics_markdown() -> str:
    """Formats test metrics and training summary as Markdown.
    Args:
        None
    Returns:
        str: Markdown block with key project metrics.
    """
    lines = [
        "### Metryki na zbiorze testowym (Lab6)",
        f"- Urządzenie inferencji: `{DEVICE}`",
        f"- Checkpoint: `{CHECKPOINT.name}`",
    ]
    if METRICS.is_file():
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        lines.extend(
            [
                f"- **Accuracy:** {metrics['accuracy']:.3f}",
                f"- **Macro-F1:** {metrics['macro_f1']:.3f}",
                f"- **ROC-AUC:** {metrics['roc_auc']:.3f}",
                f"- **PR-AUC:** {metrics['pr_auc']:.3f}",
            ]
        )
        for class_name, values in metrics.get("per_class", {}).items():
            lines.append(
                f"- `{class_name}`: P={values['precision']:.3f}, "
                f"R={values['recall']:.3f}, F1={values['f1']:.3f}"
            )
    else:
        lines.append("- Brak `metrics_test.json` — uruchom `scripts/evaluate_model.py`.")
    if HISTORY.is_file():
        history = json.loads(HISTORY.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "### Trening (Lab4)",
                f"- Best epoch: **{history.get('best_epoch')}**",
                f"- Best val loss: **{history.get('best_val_loss'):.4f}**",
                f"- Parametry trenowalne: **{history.get('trainable_params')}**",
            ]
        )
    return "\n".join(lines)


def evaluation_plot_paths() -> List[str]:
    """Collects existing evaluation/EDA plot file paths for the gallery.
    Args:
        None
    Returns:
        List[str]: Paths to PNG plots that exist on disk.
    """
    names = [
        "learning_curves.png",
        "confusion_matrix.png",
        "roc_pr.png",
        "eda_class_balance.png",
        "eda_samples.png",
    ]
    return [str(PLOTS_DIR / name) for name in names if (PLOTS_DIR / name).is_file()]


def predict_image(image: Optional[Image.Image]) -> Tuple[str, plt.Figure, Image.Image, str]:
    """Runs Softmax inference and Grad-CAM for an uploaded PIL image.
    Args:
        image (Optional[Image.Image]): User-uploaded RGB/any-mode image.
    Returns:
        Tuple[str, Figure, Image, str]: Summary text, prob chart, Grad-CAM overlay, metrics md.
    """
    if image is None:
        raise gr.Error("Najpierw wrzuć zdjęcie twarzy (JPG/PNG).")
    model = load_model()
    assert CAM_ENGINE is not None
    rgb = image.convert("RGB")
    tensor = TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs_t = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = int(probs_t.argmax().item())
    label = IDX_TO_CLASS[pred_idx]
    confidence = float(probs_t[pred_idx]) * 100.0
    probs = {
        "with_mask": float(probs_t[0]),
        "without_mask": float(probs_t[1]),
    }
    heatmap, _ = CAM_ENGINE.generate(tensor, class_idx=pred_idx)
    overlay = Image.fromarray(overlay_cam(denormalize(tensor[0]), heatmap.numpy()))
    summary = (
        f"### Werdykt modelu\n"
        f"- **Klasa:** `{label}`\n"
        f"- **Pewność:** **{confidence:.2f}%**\n"
        f"- P(with_mask) = {probs['with_mask']:.4f}\n"
        f"- P(without_mask) = {probs['without_mask']:.4f}\n"
        f"- Indeks wyjścia: `{pred_idx}` (0=with_mask, 1=without_mask)\n"
        f"- Tensor wejściowy: `[1, 3, 224, 224]` (ImageNet normalize)"
    )
    return summary, probability_figure(probs), overlay, load_metrics_markdown()


def build_app() -> gr.Blocks:
    """Constructs the Gradio Blocks UI for interactive model inspection.
    Args:
        None
    Returns:
        gr.Blocks: Configured Gradio application.
    """
    plot_paths = evaluation_plot_paths()
    with gr.Blocks(title="SNiUG — Detekcja maski") as demo:
        gr.Markdown(
            """
            # Detekcja maski ochronnej — TinyMaskCNN (from scratch)
            Wrzuć zdjęcie twarzy. Model zwróci klasę, Softmax, mapę **Grad-CAM**
            oraz pokaże wykresy z ewaluacji projektu (Lab4–Lab7).
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Zdjęcie wejściowe", height=360)
                btn = gr.Button("Uruchom inferencję", variant="primary")
            with gr.Column(scale=1):
                out_summary = gr.Markdown(label="Wynik")
                out_probs = gr.Plot(label="Prawdopodobieństwa klas")
                out_cam = gr.Image(type="pil", label="Grad-CAM (Lab7)", height=320)
        out_metrics = gr.Markdown(value=load_metrics_markdown())
        gr.Markdown("### Wykresy projektu (Lab3/Lab6)")
        gr.Gallery(
            value=plot_paths,
            label="Learning curves / Confusion matrix / ROC-PR / EDA",
            columns=3,
            height=420,
            object_fit="contain",
        )
        btn.click(fn=predict_image, inputs=inp, outputs=[out_summary, out_probs, out_cam, out_metrics])
        inp.change(fn=predict_image, inputs=inp, outputs=[out_summary, out_probs, out_cam, out_metrics])
    return demo


def main() -> None:
    """Starts the Gradio server bound for Windows browser access from WSL.
    Args:
        None
    Returns:
        None
    """
    load_model()
    print(f"device={DEVICE} checkpoint={CHECKPOINT}")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
