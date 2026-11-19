# Lab5 — Inferencja i serwowanie modelu

> **Uwaga:** folder `Lab5/` nie zawierał PDF z zajęciami. Deliverable oparty o sekcję *Serwowanie modelu (GUI/Deployment)* ze specyfikacji projektu (Lab2 SNiUG) oraz ciągłość po Lab4.

## Przepływ aplikacji
1. Wejście: plik graficzny (ścieżka CLI)
2. Preprocess: RGB → Resize 224 → ImageNet normalize → `[1,3,224,224]`
3. Inferencja: logity → Softmax
4. Wynik: etykieta + pewność (%) + opcjonalnie Grad-CAM

## Uruchomienie
```bash
python scripts/infer_image.py path/to/image.jpg --gradcam
```

Kod: `scripts/infer_image.py` (kontrakt zgodny z GUI; GUI web można dołożyć później bez zmiany preprocessu).
