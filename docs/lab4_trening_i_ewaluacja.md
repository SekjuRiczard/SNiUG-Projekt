# Lab4 — Trening modelu from scratch

## Strategia
- **From scratch:** `TinyMaskCNN`, Kaiming init, wszystkie wagi trenowalne
- Alternatywa (kolega): fine-tuning MobileNetV2 — zestawienie później

## Implementacja
- `src/model.py`, `src/data.py`, `src/train.py`
- Wejście `[B,3,224,224]`, wyjście `[B,2]`, `CrossEntropyLoss`, AdamW lr=1e-3
- Augmentacje tylko train; early stopping na val loss; checkpoint `artifacts/checkpoints/best.pt`

## Uruchomienie
```bash
python scripts/train_scratch.py
```

## Deliverable
- Skrypty + `requirements.txt`
- Historia: `artifacts/checkpoints/history.json`
- Best checkpoint wg **minimalnego val loss**
- Liczba parametrów w history JSON
