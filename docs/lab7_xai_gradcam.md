# Lab7 — Wyjaśnialność (XAI): Grad-CAM

## Dobór metody
Architektura CNN → **Grad-CAM** na ostatniej warstwie `Conv2d` (`model.features[12]`).  
Uzasadnienie: lokalizuje regiony wpływające na logit klasy (usta/nos vs tło).

## Deliverable
- Opis modelu i metody
- Mapy dla ≥3 próbek (w tym błędna, jeśli występuje)
- Porównanie z intuicją domenową + ograniczenia Grad-CAM
- Kod: `src/gradcam.py`, `scripts/run_gradcam.py`

```bash
python scripts/run_gradcam.py
```

Artefakty: `artifacts/plots/gradcam/`, `artifacts/reports/gradcam_examples.json`
