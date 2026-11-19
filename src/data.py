"""Dataset helpers, splits and transforms for face-mask classification."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLASS_TO_IDX: Dict[str, int] = {"with_mask": 0, "without_mask": 1}
IDX_TO_CLASS: Dict[int, str] = {0: "with_mask", 1: "without_mask"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SplitPaths:
    """Train/val/test file lists with integer labels."""
    train: List[Tuple[Path, int]]
    val: List[Tuple[Path, int]]
    test: List[Tuple[Path, int]]


def discover_samples(root: Path) -> Dict[str, List[Path]]:
    """Collects image paths for each class folder under root/dataset or root.
    Args:
        root (Path): Dataset root containing with_mask/without_mask folders.
    Returns:
        Dict[str, List[Path]]: Mapping class name -> list of image paths.
    """
    base = root / "dataset" if (root / "dataset").is_dir() else root
    samples: Dict[str, List[Path]] = {}
    for class_name in CLASS_TO_IDX:
        folder = base / class_name
        files = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        samples[class_name] = files
    return samples


def stratified_split(
    samples: Dict[str, List[Path]],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitPaths:
    """Creates a stratified holdout split per class.
    Args:
        samples (Dict[str, List[Path]]): Class-wise image paths.
        seed (int): RNG seed for reproducibility.
        train_ratio (float): Fraction of train samples per class.
        val_ratio (float): Fraction of validation samples per class.
    Returns:
        SplitPaths: Disjoint train/val/test path lists with labels.
    """
    rng = random.Random(seed)
    train: List[Tuple[Path, int]] = []
    val: List[Tuple[Path, int]] = []
    test: List[Tuple[Path, int]] = []
    for class_name, paths in samples.items():
        label = CLASS_TO_IDX[class_name]
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend((p, label) for p in shuffled[:n_train])
        val.extend((p, label) for p in shuffled[n_train:n_train + n_val])
        test.extend((p, label) for p in shuffled[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitPaths(train=train, val=val, test=test)


def build_transforms(train: bool) -> transforms.Compose:
    """Builds torchvision transforms for train (aug) or eval (deterministic).
    Args:
        train (bool): Whether to enable stochastic augmentations.
    Returns:
        transforms.Compose: Preprocessing pipeline to 224 ImageNet tensor.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MaskDataset(Dataset):
    """Image folder dataset returning (tensor, label) pairs."""

    def __init__(self, items: List[Tuple[Path, int]], train: bool) -> None:
        """Stores samples and selects train/eval transforms.
        Args:
            items (List[Tuple[Path, int]]): Path/label pairs.
            train (bool): Enables augmentations when True.
        Returns:
            None
        """
        self.items = items
        self.transform = build_transforms(train=train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def make_loaders(
    root: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SplitPaths]:
    """Builds stratified dataloaders for train/val/test.
    Args:
        root (Path): Dataset root path.
        batch_size (int): Mini-batch size.
        num_workers (int): DataLoader workers.
        seed (int): Split seed.
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, SplitPaths]: Loaders and split.
    """
    samples = discover_samples(root)
    split = stratified_split(samples, seed=seed)
    train_loader = DataLoader(
        MaskDataset(split.train, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        MaskDataset(split.val, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        MaskDataset(split.test, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, split
