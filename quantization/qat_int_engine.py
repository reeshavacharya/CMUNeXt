"""
qat_int_engine.py

Purpose:
    Thin wrapper around int_engine.py for running inference on artifacts
    exported from QAT.

Why separate file:
    keeps PTQ and QAT experiment flows separate and easier to compare.
"""

from __future__ import annotations

from int_engine import (
    load_quantized_artifacts,
    load_input_image_as_tensor,
    run_integer_inference,
)


def parse_args():
    """
    CLI args for QAT integer inference.
    """
    raise NotImplementedError


def main() -> None:
    """
    Run integer inference using QAT-exported artifacts.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()