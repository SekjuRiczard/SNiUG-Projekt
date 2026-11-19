# Lab1 — Karta projektu

Zgodne z *Zajęcia projektowe 1: Wybór tematu projektu i pozyskiwanie danych*.

## 1. Tytuł roboczy
**Klasyfikacja obecności maski ochronnej na twarzy (from scratch vs fine-tuning)**

Wariant realizowany w tym repozytorium: **uczenie modelu CNN od podstaw (from scratch)**.  
Wariant kolegi (osobna gałąź prac): fine-tuning MobileNetV2 — wspólne zestawienie na końcu.

## 2. Źródło danych
- **Główne (specyfikacja zespołu / Lab2):**  
  [Face Mask Dataset — Omkar Gurav (Kaggle)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  
  ~7553 obrazy RGB, klasy: `with_mask` (~3725), `without_mask` (~3828), zbiór zrównoważony.
- **Kopia lokalna / alternatywa robocza:**  
  `../Projekt/archive.zip` oraz skrót Lab1 `DataSet.url` → [susandaneshmand/mask-detection](https://www.kaggle.com/datasets/susandaneshmand/mask-detection) (~4095 obrazów w dwóch klasach).

**Decyzja na Lab1:** oficjalnym źródłem do raportowania i porównania z kolegą jest zbiór **Omkar Gurav**. Lokalny `archive.zip` może służyć do szybkich testów pipeline’u do czasu pełnego pobrania zbioru docelowego (uzgodnienie przed Lab3/Lab4).

Sposób pozyskania: **pobieranie statyczne** (gotowy, etykietowany dataset z Kaggle) — spójność etykiet i powtarzalność eksperymentów.

## 3. Co ma robić model?
Model przyjmuje **statyczny obraz twarzy (RGB)**, zwraca **prawdopodobieństwa dwóch klas** (`with_mask` / `without_mask`) oraz — w warstwie aplikacji — **etykietę z pewnością (%)** i opcjonalnie **mapę Grad-CAM** (który region obrazu wpłynął na decyzję).

Docelowy kontrakt (uszczegółowiony w Lab2):
- wejście: tensor `[B, 3, 224, 224]`,
- wyjście: logity `[B, 2]` → Softmax.

Zadanie: **klasyfikacja binarna** obrazów (nie segmentacja, nie detekcja wielu twarzy w kadrze jako osobny detektor — klasyfikacja całego obrazu / cropa twarzy ze zbioru).

## 4. Plan listy zadań (roadmap projektu)

### Lab1 — temat i dane *(ten dokument)*
- [x] Wybór tematu i strategii (from scratch)
- [x] Wskazanie źródła danych i sposobu pozyskania
- [x] Opis działania modelu
- [x] Plan prac

### Lab2 — analiza danych i dobór architektury
- [ ] Specyfikacja kształtu tensora wejściowego / wyjściowego
- [ ] Propozycja architektury CNN (from scratch) + uzasadnienie inductive bias
- [ ] Strategia uczenia: from scratch (vs fine-tuning jako alternatywa kolegi)
- [ ] Loss + metryki (CrossEntropy, F1, CM, ROC-AUC, …)

### Lab3 — walidacja i EDA
- [ ] Opis zbioru, rozkład klas, jakość danych
- [ ] Wizualizacje próbek / cech (oraz PCA/t-SNE jeśli możliwe)
- [ ] Strategia walidacji (holdout stratyfikowany Train/Val/Test)
- [ ] Zabezpieczenie przed data leakage
- [ ] Plan metryk oceny

### Lab4 — trening i ewaluacja
- [ ] Implementacja modelu from scratch w PyTorch
- [ ] Trening na GPU (WSL2 + CUDA), monitoring, early stopping / best checkpoint
- [ ] Ocena na teście + krzywe uczenia + przykłady błędów
- [ ] `requirements.txt`, raport, porównanie wstępne z wariantem fine-tuning (gdy gotowy)

### Po Lab4 — serwowanie i zestawienie
- [ ] Prosty GUI / web: upload → preprocess → Softmax + Grad-CAM
- [ ] Wspólna tabela porównawcza from scratch vs fine-tuning (metryki, #parametrów, czas, VRAM)

## 5. Kryteria wyboru tematu (Lab1)
| Kryterium | Ocena |
|-----------|--------|
| Kompletność danych | Zbiór etykietowany, dwie jasne klasy, prawie zbalansowany |
| Jasność celu | Predykcja kategorii: z maską / bez maski |
| Dostępność metryk | Etykiety „prawdy” → accuracy, F1, CM, AUC |
| Sprzęt | GTX 1650 Ti 4 GB — lekka CNN from scratch jest realna obliczeniowo |
