# SNiUG-Projekt — klasyfikacja masek (from scratch)

## Status laboratoriów
| Lab | Branch | Status |
|-----|--------|--------|
| 1 | `lab1-karta-projektu` | ✅ |
| 2 | `lab2-specyfikacja-architektury` | ✅ |
| 3 | `lab3-eda-walidacja` | ✅ |
| 4 | `lab4-trening-from-scratch` | ✅ |
| 5 | `lab5-inferencja-serwowanie` | ✅ |
| 6 | `lab6-ewaluacja-jakosci` | ⏳ |
| 7 | `lab7-xai-gradcam` | ⏳ |

## Delta Lab5 vs Lab4
**Lab4** = trening + checkpoint.  
**Lab5 dodaje** serwowanie/inferencję CLI:
- `scripts/infer_image.py` — RGB→224→normalize → Softmax → etykieta + %
- opcjonalny `--gradcam`
- docs: `docs/lab5_inferencja_serwowanie.md` (PDF Lab5 nie był w folderze — wg sekcji Deployment ze specyfikacji)

## Wynik treningu (Lab4)
- ~250k parametrów, 10 epok, best val loss ≈ 0.326
- Test (Lab6): accuracy ≈ **0.86**, macro-F1 ≈ **0.86**, ROC-AUC ≈ **0.93**

## Środowisko
WSL2 + CUDA — `docs/env_wsl_gpu.md`  
`MASK_DATA_ROOT=/home/dawid/datasets/mask`
