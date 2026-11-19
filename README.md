# SNiUG-Projekt — klasyfikacja masek (from scratch)

## Status laboratoriów
| Lab | Branch | Status |
|-----|--------|--------|
| 1–5 | … | ✅ |
| 6 | `lab6-ewaluacja-jakosci` | ✅ |
| 7 | `lab7-xai-gradcam` | ⏳ |

## Delta Lab6 vs Lab5
**Lab5** = inferencja pojedynczego obrazu.  
**Lab6 dodaje** kompleksową ewaluację na **teście**:
- krzywe uczenia train/val
- Confusion Matrix, ROC, PR
- metryki: accuracy **0.860**, macro-F1 **0.859**, ROC-AUC **0.930**, PR-AUC **0.932**
- kod: `src/evaluate.py`, `scripts/evaluate_model.py`
- docs: `docs/lab6_ewaluacja_jakosci.md`
- artefakty: `artifacts/reports/metrics_test.json`, `artifacts/plots/*.png`

## Checkpoint
Wybór według **min val loss** (epoka 10, val≈0.326), ~249 922 parametrów trenowalnych.
