"""Report whether this project can use the installed NVIDIA GPU."""
from __future__ import annotations
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    try:
        import torch
        import torchvision
        import pandas
        import sklearn
        import yaml
    except ImportError as error:
        print(f"FAILED: missing package: {error}")
        return 1
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}")
        print(f"VRAM: {properties.total_memory / 1024**3:.1f} GB")
        print("READY: GTX 1650 settings can be used.")
        return 0
    print("WARNING: CUDA is not available. Training will run on CPU and be much slower.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
