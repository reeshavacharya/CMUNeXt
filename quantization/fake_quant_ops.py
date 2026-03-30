"""
fake_quant_ops.py

Purpose:
    Reusable fake-quantization modules and QAT-aware wrappers for CMUNeXt.

Goal:
    Simulate the final integer inference path during training while keeping
    training differentiable.

Main responsibilities:
    - activation fake quant
    - weight fake quant
    - observer / range tracking
    - GELU-compatible fake quant wrapper
    - BN fake quant wrapper
    - Add / Concat alignment helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class FakeQuantConfig:
    """
    Configuration for fake quantization.

    Fields:
        dtype:
            Usually 'int8'
        symmetric:
            Whether to use symmetric quantization
        per_channel:
            Whether to use per-channel quantization
        channel_axis:
            Channel axis for per-channel quantization
        eps:
            Small constant for numerical stability
    """
    dtype: str = "int8"
    symmetric: bool = False
    per_channel: bool = False
    channel_axis: int = 1
    eps: float = 1e-8


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def get_qrange(dtype: str) -> Tuple[int, int]:
    """
    Return quantized integer range for a dtype.
    """
    if dtype == "int8":
        return -128, 127
    if dtype == "uint8":
        return 0, 255
    if dtype == "int32":
        return -(2**31), 2**31 - 1
    raise ValueError(f"Unsupported dtype: {dtype}")


def choose_qparams_tensor(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    config: FakeQuantConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scale and zero point from tensor min/max.

    Supports:
        - per-tensor affine/symmetric
        - per-channel affine/symmetric

    Args:
        x_min:
            Tensor minimum(s), scalar or per-channel tensor
        x_max:
            Tensor maximum(s), scalar or per-channel tensor
        config:
            Fake quant config

    Returns:
        (scale, zero_point)
    """
    qmin, qmax = get_qrange(config.dtype)

    if config.symmetric:
        abs_max = torch.maximum(x_min.abs(), x_max.abs())
        scale = abs_max / float(qmax)
        scale = torch.clamp(scale, min=config.eps)
        zero_point = torch.zeros_like(scale)
    else:
        scale = (x_max - x_min) / float(qmax - qmin)
        scale = torch.clamp(scale, min=config.eps)

        zero_point = qmin - torch.round(x_min / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)

    return scale, zero_point


def fake_quantize_ste(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
    channel_axis: Optional[int] = None,
) -> torch.Tensor:
    """
    Fake quantize with straight-through estimator.

    Forward:
        x -> quantize -> clamp -> dequantize

    Backward:
        behaves like identity (STE) because quant/dequant path is detached
        except for the x passthrough.

    Args:
        x:
            Input tensor
        scale:
            Scalar or per-channel scale
        zero_point:
            Scalar or per-channel zero point
        qmin, qmax:
            Integer quantization range
        channel_axis:
            If per-channel, axis that scale/zp apply to

    Returns:
        Fake-quantized tensor in float domain
    """
    if channel_axis is not None and scale.numel() > 1:
        shape = [1] * x.dim()
        shape[channel_axis] = -1
        scale = scale.view(*shape)
        zero_point = zero_point.view(*shape)

    q = torch.round(x / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    x_hat = (q - zero_point) * scale

    # Straight-through estimator
    return x + (x_hat - x).detach()


# ---------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------

class MinMaxObserver(nn.Module):
    """
    Track running min/max of tensors during QAT.

    First-pass design:
        - min/max observer
        - supports per-tensor or per-channel tracking
        - freezeable
    """

    def __init__(self, config: FakeQuantConfig) -> None:
        super().__init__()
        self.config = config
        self.enabled = True

        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

        # For per-channel observers, these buffers will be replaced lazily
        self._initialized_per_channel = False

    def _reduce_per_channel(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-channel min/max for tensor x.
        """
        channel_axis = self.config.channel_axis
        reduce_dims = [d for d in range(x.dim()) if d != channel_axis]

        x_min = x.amin(dim=reduce_dims)
        x_max = x.amax(dim=reduce_dims)
        return x_min.detach(), x_max.detach()

    def _reduce_per_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scalar min/max for tensor x.
        """
        return x.min().detach(), x.max().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Update observer stats and return x unchanged.
        """
        if not self.enabled:
            return x

        if self.config.per_channel:
            x_min, x_max = self._reduce_per_channel(x)

            if not self._initialized_per_channel:
                self.min_val = x_min.clone()
                self.max_val = x_max.clone()
                self._initialized_per_channel = True
            else:
                self.min_val = torch.minimum(self.min_val, x_min)
                self.max_val = torch.maximum(self.max_val, x_max)
        else:
            x_min, x_max = self._reduce_per_tensor(x)
            self.min_val = torch.minimum(self.min_val, x_min)
            self.max_val = torch.maximum(self.max_val, x_max)

        return x

    def get_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (scale, zero_point) from tracked stats.
        """
        min_val = self.min_val
        max_val = self.max_val

        # Handle completely uninitialized edge case
        if torch.isinf(min_val).any() or torch.isinf(max_val).any():
            if self.config.per_channel:
                # If not initialized but per-channel requested, fallback scalar
                min_val = torch.zeros(1, device=min_val.device)
                max_val = torch.ones(1, device=max_val.device)
            else:
                min_val = torch.tensor(0.0, device=min_val.device)
                max_val = torch.tensor(1.0, device=max_val.device)

        return choose_qparams_tensor(min_val, max_val, self.config)

    def freeze(self) -> None:
        """
        Stop updating observer stats.
        """
        self.enabled = False


# ---------------------------------------------------------------------
# Fake-quant modules
# ---------------------------------------------------------------------

class FakeQuantizeSTE(nn.Module):
    """
    Straight-through-estimator fake quant module.

    Uses observer-tracked qparams unless observers are frozen.
    """

    def __init__(self, config: FakeQuantConfig) -> None:
        super().__init__()
        self.config = config
        self.observer = MinMaxObserver(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fake quantize then dequantize x.
        """
        x = self.observer(x)
        scale, zero_point = self.observer.get_qparams()
        qmin, qmax = get_qrange(self.config.dtype)

        channel_axis = self.config.channel_axis if self.config.per_channel else None
        return fake_quantize_ste(
            x=x,
            scale=scale,
            zero_point=zero_point,
            qmin=qmin,
            qmax=qmax,
            channel_axis=channel_axis,
        )

    def get_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expose current scale and zero point.
        """
        return self.observer.get_qparams()

    def freeze_observer(self) -> None:
        self.observer.freeze()


class WeightFakeQuant(nn.Module):
    """
    Fake quantization for Conv weights.

    Typical config:
        - int8
        - symmetric
        - per-output-channel
    """

    def __init__(self, config: FakeQuantConfig) -> None:
        super().__init__()
        self.config = config
        self.fake_quant = FakeQuantizeSTE(config)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.fake_quant(w)

    def get_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fake_quant.get_qparams()

    def freeze_observer(self) -> None:
        self.fake_quant.freeze_observer()


class ActivationFakeQuant(nn.Module):
    """
    Fake quantization for activations.

    Typical config:
        - int8
        - affine
        - per-tensor
    """

    def __init__(self, config: FakeQuantConfig) -> None:
        super().__init__()
        self.config = config
        self.fake_quant = FakeQuantizeSTE(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fake_quant(x)

    def get_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fake_quant.get_qparams()

    def freeze_observer(self) -> None:
        self.fake_quant.freeze_observer()


# ---------------------------------------------------------------------
# QAT wrappers
# ---------------------------------------------------------------------

class QATConv2d(nn.Module):
    """
    QAT wrapper for Conv2d.

    First-pass simulation:
        input fake quant -> weight fake quant -> conv -> output fake quant
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        act_config: FakeQuantConfig,
        weight_config: FakeQuantConfig,
    ) -> None:
        super().__init__()

        # Keep original module parameters/structure
        self.conv = conv

        self.input_fake_quant = ActivationFakeQuant(act_config)
        self.weight_fake_quant = WeightFakeQuant(weight_config)
        self.output_fake_quant = ActivationFakeQuant(act_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.input_fake_quant(x)
        w_q = self.weight_fake_quant(self.conv.weight)

        y = nn.functional.conv2d(
            x_q,
            w_q,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

        y_q = self.output_fake_quant(y)
        return y_q


class QATBatchNorm2d(nn.Module):
    """
    QAT wrapper for unfused BatchNorm2d.

    First-pass design:
        input fake quant -> BN -> output fake quant

    This preserves architecture and simulates quantization boundaries around BN.
    """

    def __init__(
        self,
        bn: nn.BatchNorm2d,
        act_config: FakeQuantConfig,
    ) -> None:
        super().__init__()
        self.bn = bn
        self.input_fake_quant = ActivationFakeQuant(act_config)
        self.output_fake_quant = ActivationFakeQuant(act_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.input_fake_quant(x)
        y = self.bn(x_q)
        y_q = self.output_fake_quant(y)
        return y_q


class QATGELU(nn.Module):
    """
    QAT wrapper for GELU compatible with later LUT export.

    First-pass design:
        input fake quant -> GELU -> output fake quant

    This helps the model adapt to quantization around GELU, while export later
    can still build an integer LUT for the final runtime.
    """

    def __init__(
        self,
        gelu: nn.GELU,
        act_config: FakeQuantConfig,
    ) -> None:
        super().__init__()
        self.gelu = gelu
        self.input_fake_quant = ActivationFakeQuant(act_config)
        self.output_fake_quant = ActivationFakeQuant(act_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.input_fake_quant(x)
        y = self.gelu(x_q)
        y_q = self.output_fake_quant(y)
        return y_q


class QATAdd(nn.Module):
    """
    QAT helper for explicit Add behavior.

    First-pass policy:
        - fake-quant each input separately
        - add them
        - fake-quant output

    This assumes both branches should live in a compatible domain.
    """

    def __init__(self, act_config: FakeQuantConfig) -> None:
        super().__init__()
        self.a_fake_quant = ActivationFakeQuant(act_config)
        self.b_fake_quant = ActivationFakeQuant(act_config)
        self.output_fake_quant = ActivationFakeQuant(act_config)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_q = self.a_fake_quant(a)
        b_q = self.b_fake_quant(b)
        y = a_q + b_q
        y_q = self.output_fake_quant(y)
        return y_q


class QATConcat(nn.Module):
    """
    QAT helper for explicit Concat behavior.

    First-pass policy:
        - fake-quant each input
        - concat
        - fake-quant output
    """

    def __init__(self, act_config: FakeQuantConfig) -> None:
        super().__init__()
        self.input_fake_quant = ActivationFakeQuant(act_config)
        self.output_fake_quant = ActivationFakeQuant(act_config)

    def forward(self, tensors: list[torch.Tensor], dim: int = 1) -> torch.Tensor:
        q_tensors = [self.input_fake_quant(t) for t in tensors]
        y = torch.cat(q_tensors, dim=dim)
        y_q = self.output_fake_quant(y)
        return y_q


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def freeze_all_observers(module: nn.Module) -> None:
    """
    Freeze observer updates recursively across model.

    Call this during late-stage QAT fine-tuning once ranges have stabilized.
    """
    for child in module.modules():
        if hasattr(child, "freeze_observer") and callable(child.freeze_observer):
            child.freeze_observer()
        if hasattr(child, "observer") and hasattr(child.observer, "freeze"):
            child.observer.freeze()


def collect_fake_quant_modules(module: nn.Module) -> Dict[str, nn.Module]:
    """
    Return all fake-quant related modules by name for debugging/export.
    """
    collected: Dict[str, nn.Module] = {}
    for name, child in module.named_modules():
        if isinstance(
            child,
            (
                FakeQuantizeSTE,
                WeightFakeQuant,
                ActivationFakeQuant,
                QATConv2d,
                QATBatchNorm2d,
                QATGELU,
                QATAdd,
                QATConcat,
            ),
        ):
            collected[name] = child
    return collected