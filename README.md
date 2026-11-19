# SNiUG-Projekt — klasyfikacja masek (from scratch)

## Status laboratoriów
| Lab | Branch | Status |
|-----|--------|--------|
| 1 | `lab1-karta-projektu` | ✅ |
| 2 | `lab2-specyfikacja-architektury` | ✅ |
| 3 | `lab3-eda-walidacja` | ✅ |
| 4 | `lab4-trening-from-scratch` | ✅ |
| 5 | `lab5-inferencja-serwowanie` | ✅ |
| 6 | `lab6-ewaluacja-jakosci` | ✅ |
| 7 | `lab7-xai-gradcam` | ✅ |

## Delta Lab7 vs Lab6
**Lab6** = metryki i wykresy jakości.  
**Lab7 dodaje** XAI Grad-CAM:
- mapa aktywacji z ostatniego Conv (`features[12]`)
- ≥3 przykłady (w tym błędne predykcje)
- `scripts/run_gradcam.py`, rozszerzenie `src/gradcam.py`
- docs: `docs/lab7_xai_gradcam.md`
- artefakty: `artifacts/plots/gradcam/`, `artifacts/reports/gradcam_examples.json`

## Skrót wyników
| Metryka test | Wartość |
|--------------|---------|
| Accuracy | 0.860 |
| Macro-F1 | 0.859 |
| ROC-AUC | 0.930 |
| PR-AUC | 0.932 |
| Parametry | 249 922 (wszystkie trenowalne, from scratch) |

## Uruchomienie (WSL)
```bash
export MASK_DATA_ROOT=/home/dawid/datasets/mask
python scripts/train_scratch.py
python scripts/evaluate_model.py
python scripts/run_gradcam.py
python scripts/infer_image.py path.jpg --gradcam
```

Repo: https://github.com/SekjuRiczard/SNiUG-Projekt
