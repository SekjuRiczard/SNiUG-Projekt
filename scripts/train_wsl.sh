#!/usr/bin/env bash
set -euo pipefail
export MASK_DATA_ROOT=/home/dawid/datasets/mask
export PYTHONUNBUFFERED=1
REPO="/mnt/c/Users/Dawid/Desktop/Studia/Sieci neuronowe i uczenie glebokie PROJEKT/SNiUG-Projekt"
cd "$REPO"
/home/dawid/miniconda3/envs/snug/bin/python -u scripts/train_scratch.py
