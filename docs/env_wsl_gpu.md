# Środowisko WSL2 + GPU (PyTorch CUDA)

## Status (zweryfikowane)
- Distro: **Ubuntu 26.04** (WSL2)
- Użytkownik domyślny: `dawid`
- GPU w WSL: `nvidia-smi` → **NVIDIA GeForce GTX 1650 Ti** (4 GB)
- Env Conda: `snug` (Python **3.11**)
- PyTorch: **2.6.0+cu124**, `torch.cuda.is_available() == True`
- Test: `scripts/check_gpu.py` (matmul na CUDA OK)

## Uruchomienie
```bash
wsl -d Ubuntu
# bashrc automatycznie: conda activate snug
python /mnt/c/Users/Dawid/Desktop/Studia/Sieci\ neuronowe\ i\ uczenie\ glebokie\ PROJEKT/SNiUG-Projekt/scripts/check_gpu.py
```

Albo bez interaktywnego shella:
```powershell
wsl -d Ubuntu -u dawid -- /home/dawid/miniconda3/envs/snug/bin/python "/mnt/c/Users/Dawid/Desktop/Studia/Sieci neuronowe i uczenie glebokie PROJEKT/SNiUG-Projekt/scripts/check_gpu.py"
```

## Zainstalowane pakiety (snug)
- `torch`, `torchvision`, `torchaudio` (CUDA 12.4 wheels)
- `numpy`, `pillow`, `scikit-learn`, `matplotlib`, `tqdm`, `opencv-python-headless`

## Uwagi
- Systemowy Python Ubuntu 26.04 to 3.14 — za nowy pod PyTorch, stąd Miniconda + 3.11.
- Miniconda: `/home/dawid/miniconda3`, env: `/home/dawid/miniconda3/envs/snug`
- Przy 4 GB VRAM trzymaj małe batche (16–32) i lekką CNN from scratch.
