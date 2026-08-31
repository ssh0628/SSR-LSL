"""Run the CIFAR-10 SSR experiment defined in setting/config.py."""

from setting.config import CONFIG
from ssr.engine import run


if __name__ == "__main__":
    run(CONFIG)
