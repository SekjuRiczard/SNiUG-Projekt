#!/usr/bin/env bash
set -euo pipefail
SRC="/mnt/c/Users/Dawid/Desktop/Studia/Sieci neuronowe i uczenie glebokie PROJEKT/SNiUG-Projekt/data/raw"
DST="/home/dawid/datasets/mask"
mkdir -p "$DST"
cp -a "$SRC/." "$DST/"
find "$DST" -type f | wc -l
ls "$DST/dataset" || ls "$DST"
