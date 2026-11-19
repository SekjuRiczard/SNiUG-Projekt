#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '[user]' 'default=dawid' > /etc/wsl.conf
cat /etc/wsl.conf
if ! grep -q 'conda shell.bash hook' /home/dawid/.bashrc; then
  printf '%s\n' 'eval "$(/home/dawid/miniconda3/bin/conda shell.bash hook)"' 'conda activate snug' >> /home/dawid/.bashrc
fi
tail -5 /home/dawid/.bashrc
