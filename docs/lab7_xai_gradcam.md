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

## Wnioski z obserwacji Grad-CAM
Trzeba weryfikować, czy sieć uczy się na **pożądanych** elementach obrazu (maska / usta / nos), a nie na **niepożądanych** (tło, ściana, ubranie itd.).  
Mimo dobrego accuracy zdarzają się zdjęcia, na których Grad-CAM **nie siedzi na masce**, tylko aktywuje tło lub inne obiekty.  
Wysoki Softmax nie gwarantuje poprawnej cechy decyzyjnej — XAI jest obowiązkowym uzupełnieniem metryk z Lab6.
