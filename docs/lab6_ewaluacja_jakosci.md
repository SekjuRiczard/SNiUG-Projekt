# Lab6 — Kompleksowa ewaluacja jakości modelu

## Protokół
- Trening/val do wyboru checkpointu (min val loss)
- **Test** tylko do końcowej oceny (bez tuningu)
- Metryki: Accuracy, macro-F1, Precision/Recall per class, ROC-AUC, PR-AUC, Confusion Matrix

## Artefakty
- `artifacts/checkpoints/best.pt`, `history.json`
- `artifacts/plots/learning_curves.png`, `confusion_matrix.png`, `roc_pr.png`
- `artifacts/reports/metrics_test.json`

```bash
python scripts/evaluate_model.py
```

## Analiza błędów
Przykłady FP/FN oraz Grad-CAM z Lab7; priorytet domenowy: unikać `without_mask → with_mask`.
