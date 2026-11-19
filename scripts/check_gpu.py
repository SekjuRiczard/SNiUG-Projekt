"""Quick GPU sanity check for WSL + PyTorch CUDA."""
import torch


def main() -> None:
    """Print CUDA availability and run a small GPU matmul.
    Args:
        None
    Returns:
        None
    """
    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"cuda_available={available}")
    print(f"device_count={torch.cuda.device_count()}")
    if not available:
        raise SystemExit(1)
    print(f"device={torch.cuda.get_device_name(0)}")
    left = torch.randn(2048, 2048, device="cuda")
    right = torch.randn(2048, 2048, device="cuda")
    product = left @ right
    torch.cuda.synchronize()
    print(f"matmul_mean={float(product.mean()):.6f}")
    print(f"vram_alloc_mb={torch.cuda.memory_allocated() / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
