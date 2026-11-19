# Lab3 — Raport EDA i plan walidacji

Zgodne z *Zajęcia projektowe 3*.

## 1. Opis zbioru
| Pole | Wartość |
|------|---------|
| Modalność | Obrazy RGB |
| Źródło robocze | `data/raw/dataset` (archive.zip / mask-detection) |
| Klasy | `with_mask` (0), `without_mask` (1) |
| Tensor wejściowy | `[B,3,224,224]` po Resize+Normalize |
| Docelowy zbiór porównawczy | Omkar Gurav (~7553) — uzgodnić z kolegą |

Liczebności lokalne (EDA): **with_mask=2164**, **without_mask=1930**, łącznie **4094**, 0 plików uszkodzonych.  
Split 70/15/15: train=2865, val=613, test=616. Szczegóły: `artifacts/reports/eda_summary.json`.

## 2. Rozkład etykiet
Zbiór lokalny jest **bliski balansowi** (~53% / ~47%). Resampling nie jest konieczny; opcjonalnie `class_weight` w CE jeśli pojawi się większa nierównowaga.

Wykres: `artifacts/plots/eda_class_balance.png`.

## 3. Jakość danych
- Sprawdzenie otwieralności PIL (`verify`) — lista uszkodzonych w JSON.
- Duplikaty nazw: brak wymuszonej deduplikacji hashami na tym etapie; przed Lab4 warto unikać przecieku przez wspólne pliki w train/test (split stratyfikowany po ścieżkach, rozłączny).
- Próbki nietypowe (np. maska pod nosem, wiele twarzy): zostawiamy — to realistyczne przypadki trudne, nie błędy pomiaru.

## 4. Wizualizacja
- Surowe próbki: `artifacts/plots/eda_samples.png`
- Po preprocessie 224+ImageNet — w Lab4 podczas treningu / Grad-CAM.

## 5. Strategia walidacji
**Holdout stratyfikowany** 70% / 15% / 15% (train/val/test), `seed=42`, osobno w każdej klasie.
- Uzasadnienie: ~4k próbek — pojedynczy holdout wystarcza; pełne K-fold byłoby kosztowne przy CNN+GPU.
- Brak podziału grupowego po osobie (dataset nie ma ID tożsamości) — ryzyko korelacji twarzy między splitami; ograniczenie opisane w wnioskach.
- Augmentacje **tylko** na train; val/test deterministic Resize+Normalize.

## 6. Anti-leakage
- Split przed fitowaniem jakichkolwiek statystyk poza stałymi ImageNet mean/std.
- Brak imputacji/selektorów cech uczących się na całym zbiorze.
- Test używany dopiero w Lab6 (końcowa ewaluacja).

## 7. Plan metryk
Accuracy, macro-F1, Precision/Recall per class, Confusion Matrix, ROC-AUC, PR-AUC.  
Koszt domenowy: FN `without_mask`→`with_mask` szczególnie istotny.

## 8. Wnioski projektowe
- CNN from scratch + lekka architektura pasuje do zrównoważonego zbioru obrazów.
- Holdout 70/15/15 + early stopping na val loss.
- Przed porównaniem z fine-tuningiem kolegi ujednolicić dataset (Omkar vs lokalny archive).
