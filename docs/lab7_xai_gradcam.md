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
Mimo dobrego accuracy modelu zdarzają się zdjęcia, na których Grad-CAM **nie rejestruje poprawnie siatki uwagi na masce**, tylko aktywuje **tło / ubranie / inne elementy kadru**.  
To ograniczenie uczenia od zera: wysoki Softmax nie zawsze oznacza, że sieć „patrzy” na cechę domenową. Dlatego XAI jest obowiązkowym uzupełnieniem tabeli metryk (Lab6).
