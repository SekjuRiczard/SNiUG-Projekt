#!/usr/bin/env bash
set -euo pipefail
export MASK_DATA_ROOT=/home/dawid/datasets/mask
export PYTHONUNBUFFERED=1
REPO="/mnt/c/Users/Dawid/Desktop/Studia/Sieci neuronowe i uczenie glebokie PROJEKT/SNiUG-Projekt"
cd "$REPO"
/home/dawid/miniconda3/envs/snug/bin/python -u scripts/evaluate_model.py
/home/dawid/miniconda3/envs/snug/bin/python -u scripts/run_gradcam.py
# smoke inference on one test image
IMG=$(find "$MASK_DATA_ROOT" -type f \( -iname '*.jpg' -o -iname '*.png' \) | head -n 1)
/home/dawid/miniconda3/envs/snug/bin/python -u scripts/infer_image.py "$IMG" --gradcam
