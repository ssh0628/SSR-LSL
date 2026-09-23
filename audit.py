"""Validate dataset arrays and decode images without changing source files."""

from dataclasses import replace

from setting.config import CONFIG
from setting.data import inspect_dataset


def main() -> None:
    inspect_dataset(
        replace(CONFIG.data, verify_images=True),
        report_path=CONFIG.runtime.output_root / "data_audit.jsonl",
    )


if __name__ == "__main__":
    main()
