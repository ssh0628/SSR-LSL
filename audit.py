"""Decode and validate the configured dataset without constructing a model."""

from dataclasses import replace

from setting.config import CONFIG
from setting.data import prepare_dataset


def main() -> None:
    prepare_dataset(
        replace(CONFIG.data, verify_images=True),
        report_path=CONFIG.runtime.output_root / "data_audit.jsonl",
    )


if __name__ == "__main__":
    main()
