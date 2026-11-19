# SNiUG-Projekt — klasyfikacja obecności maski ochronnej

Projekt z przedmiotu **Sieci neuronowe i uczenie głębokie** (Akademia Tarnowska).

## Cel
Klasyfikacja binarna obrazów twarzy: `with_mask` / `without_mask`, z prawdopodobieństwem Softmax oraz mapą Grad-CAM w aplikacji serwującej model.

## Dwa podejścia (zestawienie)
| Gałąź prac | Strategia | Opis |
|------------|-----------|------|
| **Ten repozytorium (from scratch)** | Uczenie od zera | `TinyMaskCNN`, losowa inicjalizacja wag |
| Kolega | Fine-tuning | MobileNetV2 (ImageNet) + nowa głowa klasyfikatora |

## Status laboratoriów
| Lab | Branch | Status |
|-----|--------|--------|
| 1 | `lab1-karta-projektu` | ✅ zmergowane |
| 2 | `lab2-specyfikacja-architektury` | ✅ w toku / merge |
| 3 | `lab3-eda-walidacja` | ⏳ |
| 4 | `lab4-trening-from-scratch` | ⏳ |
| 5 | `lab5-inferencja-serwowanie` | ⏳ (PDF Lab5 brak w folderze — Deployment ze specyfikacji) |
| 6 | `lab6-ewaluacja-jakosci` | ⏳ |
| 7 | `lab7-xai-gradcam` | ⏳ |

## Delta względem poprzedniego labu (Lab2 vs Lab1)
**Lab1** miało kartę: temat, źródło danych, plan, opis „co robi model”.  
**Lab2 dodaje:**
- kontrakt tensorów `[B,3,224,224]` → `[B,2]`
- architekturę **TinyMaskCNN** (from scratch) + uzasadnienie CNN
- strategię uczenia (AdamW, CE, augmentacje, early stopping)
- listę metryk i zapowiedź Grad-CAM  
Plik: [`docs/lab2_specyfikacja_architektury.md`](docs/lab2_specyfikacja_architektury.md)

## Kontrakt I/O
- **Wejście:** `[B, 3, 224, 224]` RGB + ImageNet normalize
- **Wyjście:** `[B, 2]` logity → Softmax (`0=with_mask`, `1=without_mask`)
- **Loss:** CrossEntropyLoss
- **Metryki:** Precision, Recall, F1, CM, ROC-AUC, PR-AUC

## Workflow git
Po każdym labie: merge do `main`, **branch zostaje**. Przyrostowo — bez wyprzedzania wymagań kolejnego labu.

## Środowisko
WSL2 + PyTorch CUDA na GTX 1650 Ti — [`docs/env_wsl_gpu.md`](docs/env_wsl_gpu.md)
