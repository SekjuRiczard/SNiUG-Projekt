#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO="/mnt/c/Users/Dawid/Desktop/Studia/Sieci neuronowe i uczenie glebokie PROJEKT/SNiUG-Projekt"
cd "$REPO"
echo "Otworz w przegladarce Windows: http://localhost:7860"
/home/dawid/miniconda3/envs/snug/bin/python -u scripts/app_gradio.py
