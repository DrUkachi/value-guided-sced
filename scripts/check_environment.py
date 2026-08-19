"""Validate dependencies and configuration without running an experiment."""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "base.yaml"

IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "PyTorch": "torch",
    "PyYAML": "yaml",
    "Matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pytest": "pytest",
}


def main() -> None:
    missing = []
    for package, module in IMPORTS.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit("Missing required packages: " + ", ".join(missing))

    import numpy as np
    import torch
    import yaml

    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for relative_path in ("results/raw", "results/processed", "results/figures"):
        (ROOT / relative_path).mkdir(parents=True, exist_ok=True)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Configuration: {CONFIG_PATH}")
    print(yaml.safe_dump(config, sort_keys=False))
    print("Environment check passed.")


if __name__ == "__main__":
    main()
