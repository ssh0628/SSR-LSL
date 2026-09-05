"""Run the CIFAR-10 experiment defined in cifar/setting/config.py."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    # Support both python -m cifar.cifar_ssr and python cifar/cifar_ssr.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar.setting.config import CONFIG
from cifar.ssr.engine import run


if __name__ == "__main__":
    run(CONFIG)
