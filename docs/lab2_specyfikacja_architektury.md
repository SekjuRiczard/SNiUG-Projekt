# Lab2 — Specyfikacja architektury projektu

Zgodne z *Zajęcia projektowe 2* oraz kartą `Lab2 SNiUG Projekt.pdf` (kontrakt I/O wspólny z wariantem fine-tuning kolegi).

## 1. Problem i dane
- Klasyfikacja binarna obrazów twarzy: `with_mask` (0) / `without_mask` (1)
- Modalność: obrazy RGB
- Strategia **tego repo**: **from scratch** (losowa inicjalizacja wszystkich wag)
- Alternatywa (kolega): fine-tuning MobileNetV2 ImageNet — ten sam kontrakt I/O i metryki

## 2. Kształty tensorów
| Etap | Kształt | Znaczenie osi |
|------|---------|----------------|
| Wejście | `[B, 3, 224, 224]` | Batch, RGB, H, W (NCHW PyTorch) |
| Preprocess | Resize 224 + ImageNet `mean/std` | jak w specyfikacji zespołu |
| Wyjście (logity) | `[B, 2]` | `0=with_mask`, `1=without_mask` |
| Inferencja | Softmax → `[B, 2]` probs | etykieta = argmax |

Batch treningowy docelowy na GTX 1650 Ti 4 GB: **B=16** (ew. 32 przy Tiny CNN).

## 3. Architektura from scratch: `TinyMaskCNN`
Rodzina: **CNN 2D** (inductive bias: lokalność + translation equivariance/invariance przez pooling).

```
Conv(3→32, 3×3, pad1) → BN → ReLU → MaxPool(2)
Conv(32→64, 3×3, pad1) → BN → ReLU → MaxPool(2)
Conv(64→128, 3×3, pad1) → BN → ReLU → MaxPool(2)
Conv(128→128, 3×3, pad1) → BN → ReLU → AdaptiveAvgPool(1×1)
Flatten → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→2)
```

Uzasadnienie:
- Maskę rozpoznaje się po lokalnych wzorcach (usta/nos) — CNN naturalnie to wyciąga.
- Lekki model (~kilkaset tys. parametrów) mieści się w 4 GB VRAM i uczy się od zera na ~4–7.5k obrazach szybciej niż ciężki backbone bez pretrained.
- Wspólna głowa 2-logitowa jak u kolegi → porównywalne Softmax/Grad-CAM.

## 4. Strategia uczenia
| Element | Wartość |
|---------|---------|
| Inicjalizacja | losowa (Kaiming dla Conv/Linear) |
| Loss | `CrossEntropyLoss` |
| Optymalizator | AdamW, lr=1e-3, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau (val loss) |
| Early stopping | patience=5 na val loss |
| Augmentacje (tylko train) | HFlip, Rotation±15°, ColorJitter |

## 5. Metryki
Precision, Recall, F1 (per klasa + macro), Confusion Matrix, ROC-AUC, PR-AUC.  
Priorytet domenowy: unikać FN „bez maski → z maską” (osoba bez maski uznana za zamaskowaną).

## 6. XAI (zapowiedź Lab7)
Grad-CAM na ostatniej warstwie konwolucyjnej (`features` / ostatni Conv).
