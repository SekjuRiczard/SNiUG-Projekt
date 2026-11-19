# SNiUG-Projekt — klasyfikacja obecności maski ochronnej

## Status laboratoriów
| Lab | Branch | Status |
|-----|--------|--------|
| 1 | `lab1-karta-projektu` | ✅ |
| 2 | `lab2-specyfikacja-architektury` | ✅ |
| 3 | `lab3-eda-walidacja` | ✅ (ten branch / merge) |
| 4–7 | `lab4`…`lab7` | ⏳ |

## Delta Lab3 vs Lab2
**Lab2** zamroziło kontrakt I/O i architekturę TinyMaskCNN.  
**Lab3 dodaje:**
- EDA: liczebności klas, jakość plików, przykładowe obrazy
- plan holdout 70/15/15 stratyfikowany (`seed=42`)
- anti-leakage + metryki oceny
- kod: `src/data.py`, `scripts/run_eda.py`
- artefakty: `artifacts/reports/eda_summary.json`, `artifacts/plots/eda_*.png`

Docs: [`docs/lab3_eda_i_walidacja.md`](docs/lab3_eda_i_walidacja.md)

## Kontrakt I/O
`[B,3,224,224]` → logity `[B,2]` → Softmax (`with_mask`/`without_mask`)

## Środowisko
WSL2 + PyTorch CUDA — [`docs/env_wsl_gpu.md`](docs/env_wsl_gpu.md)
