# SNiUG-Projekt — klasyfikacja obecności maski ochronnej

Projekt z przedmiotu **Sieci neuronowe i uczenie głębokie** (Akademia Tarnowska).

## Cel
Klasyfikacja binarna obrazów twarzy: `with_mask` / `without_mask`, z prawdopodobieństwem Softmax oraz mapą Grad-CAM w aplikacji serwującej model.

## Dwa podejścia (zestawienie)
| Gałąź prac | Strategia | Opis |
|------------|-----------|------|
| **Ten repozytorium (from scratch)** | Uczenie od zera | Własna / lekka CNN, losowa inicjalizacja wag |
| Kolega | Fine-tuning | MobileNetV2 (ImageNet) + nowa głowa klasyfikatora |

Wspólny kontrakt I/O, ten sam podział danych i te same metryki — żeby porównanie było miarodajne.

## Kontrakt I/O (docelowy)
- **Wejście:** `[B, 3, 224, 224]` RGB, Resize 224, normalizacja ImageNet
- **Wyjście:** `[B, 2]` logity → Softmax (`0=with_mask`, `1=without_mask`)
- **Loss:** CrossEntropyLoss
- **Metryki:** Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR-AUC

## Struktura repozytorium
```
docs/          # Deliverables per laboratorium projektowe
src/           # Kod (pojawia się od Lab4 / wcześniej jeśli potrzeba skryptów EDA)
data/          # Opis danych / ścieżki (same obrazy poza gitem lub LFS)
artifacts/     # Checkpointy, wykresy (niecommitowane duże pliki)
```

## Workflow git (branche zadaniowe)
| Branch | Laboratorium | Deliverable |
|--------|--------------|-------------|
| `lab1-karta-projektu` | Lab1 | Karta projektu: tytuł, źródło danych, plan zadań, opis modelu |
| `lab2-specyfikacja-architektury` | Lab2 | Specyfikacja: kształty tensorów, architektura, strategia, loss/metryki |
| `lab3-eda-walidacja` | Lab3 | Raport EDA + plan walidacji |
| `lab4-trening-from-scratch` | Lab4 | Trening from scratch, ewaluacja, checkpoint, raport |

Zasady:
1. Praca wyłącznie na branchu danego labu.
2. Po akceptacji wymagań labu → **merge do `main`**.
3. **Branche zadaniowe zostają** (nie usuwamy ich po merge).
4. Na danym etapie w `main` jest tylko to, czego wymaga ukończony konspekt (przyrostowo).

## Dataset
- Specyfikacja zespołu: [Face Mask Dataset (Omkar Gurav, Kaggle)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Lokalna kopia robocza może być w katalogu obok repo (`../Projekt/archive.zip`) — szczegóły w `docs/`.

## Środowisko
Trening na GPU (NVIDIA GTX 1650 Ti) w **WSL2 + PyTorch CUDA**. Szczegóły instalacji: `docs/env_wsl_gpu.md` (uzupełniane wraz z Lab4 / setupem).
