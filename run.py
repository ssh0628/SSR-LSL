"""Run the image-dataset experiment defined in setting/config.py."""

from setting.config import CONFIG
from ssr.engine import run


def main() -> None:
    run(CONFIG)


if __name__ == "__main__":
    main()
