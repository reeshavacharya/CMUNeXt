"""
qat_utils.py

Purpose:
    Shared helper functions for QAT pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def save_json(obj: Any, path: str) -> None:
    """
    Save JSON-serializable object.
    """
    raise NotImplementedError


def load_json(path: str) -> Any:
    """
    Load JSON object.
    """
    raise NotImplementedError


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move tensor values in a batch dict to the target device.
    """
    raise NotImplementedError


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Set requires_grad recursively.
    """
    raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable + non-trainable parameters.
    """
    raise NotImplementedError


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Save QAT checkpoint.
    """
    raise NotImplementedError


def load_checkpoint(
    model: nn.Module,
    path: str,
    map_location: str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    """
    Load QAT checkpoint into model (and optionally optimizer).

    Returns:
        metadata dict including epoch if available.
    """
    raise NotImplementedError


def make_output_dir(path: str) -> None:
    """
    Create output directory if needed.
    """
    raise NotImplementedError