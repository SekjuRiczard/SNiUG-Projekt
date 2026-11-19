# SNiUG-Projekt — klasyfikacja obecności maski ochronnej

## Status laboratoriów
| Lab | Branch | Status |
|-----|--------|--------|
| 1 | `lab1-karta-projektu` | ✅ |
| 2 | `lab2-specyfikacja-architektury` | ✅ |
| 3 | `lab3-eda-walidacja` | ✅ |
| 4 | `lab4-trening-from-scratch` | ✅ (ten branch / merge) |
| 5 | `lab5-inferencja-serwowanie` | ⏳ |
| 6 | `lab6-ewaluacja-jakosci` | ⏳ |
| 7 | `lab7-xai-gradcam` | ⏳ |

## Delta Lab4 vs Lab3
**Lab3** dało EDA, split i `src/data.py`.  
**Lab4 dodaje:**
- `TinyMaskCNN` (`src/model.py`) + pętla treningu (`src/train.py`)
- `scripts/train_scratch.py`, `requirements.txt`
- AdamW + ReduceLROnPlateau + early stopping, checkpoint `artifacts/checkpoints/best.pt`
- dokumentacja: `docs/lab4_trening_i_ewaluacja.md`

## Uruchomienie treningu (WSL)
```bash
export MASK_DATA_ROOT=/home/dawid/datasets/mask   # szybki dysk Linux
python scripts/train_scratch.py
```

## Kontrakt I/O
`[B,3,224,224]` → `[B,2]` Softmax
