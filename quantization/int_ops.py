"""
int_ops.py

Purpose:
    Integer arithmetic kernels for CMUNeXt PTQ inference.

Main responsibilities:
        - integer Conv2d
        - integer requantization
        - integer unfused BatchNorm2d
        - GELU LUT execution
        - maxpool
        - nearest upsample
        - concat
        - add

Important:
    - All runtime arithmetic here must match the conventions used in quantize.py.
    - quantize.py computes constants; int_ops.py consumes them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def clamp_int_tensor(
    x: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """
    Clamp integer tensor to quantized range.
    """
    return torch.clamp(x, qmin, qmax)


def round_shift_right(
    x: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """
    Apply shift with explicit rounding.

    Convention:
        - shift > 0: arithmetic right shift with sign-aware rounding
        - shift == 0: unchanged
        - shift < 0: left shift by -shift
    """
    if shift == 0:
        return x
    elif shift < 0:
        return x << (-shift)

    # shift > 0
    bias = 1 << (shift - 1)
    return torch.where(
        x >= 0,
        (x + bias) >> shift,
        (x - bias) >> shift,
    )


def requantize_int32(
    acc_int32: torch.Tensor,
    multiplier_int: int,
    shift: int,
    out_zero_point: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """
    Requantize int32 accumulator into output quantized domain.

    Compatible with quantize_multiplier() where:
        real_multiplier ≈ (multiplier_int / 2^31) * 2^shift

    Runtime:
        x_scaled = round(acc * multiplier_int / 2^31)
        if shift > 0:
            x_scaled = x_scaled << shift
        elif shift < 0:
            x_scaled = rounded_right_shift(x_scaled, -shift)

    Returns:
        Int32 tensor clamped into [qmin, qmax]
    """
    if acc_int32.dtype != torch.int32:
        acc_int32 = acc_int32.to(torch.int32)

    x = acc_int32.to(torch.int64) * int(multiplier_int)

    # First remove Q31 fractional scaling
    x = round_shift_right(x, 31)

    # Then apply exported power-of-two shift
    if shift > 0:
        x = x << shift
    elif shift < 0:
        x = round_shift_right(x, -shift)

    x = x + int(out_zero_point)
    x = torch.clamp(x, int(qmin), int(qmax))

    return x.to(torch.int32)

def extract_patches_int64(
    q_input_i64: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> torch.Tensor:
    """
    Extract convolution patches in pure integer arithmetic.

    Args:
        q_input_i64:
            Input tensor [N, C, H, W] in int64
        kernel_size:
            (kH, kW)
        stride:
            (sH, sW)
        padding:
            (pH, pW)
        dilation:
            (dH, dW)

    Returns:
        patches:
            Tensor of shape [N, C * kH * kW, L], where L = H_out * W_out

    Notes:
        - This is an integer-safe replacement for F.unfold, which is not
          implemented for Long tensors on CPU.
        - It uses explicit slicing over the padded input.
    """
    if q_input_i64.dim() != 4:
        raise ValueError(f"q_input_i64 must be 4D [N,C,H,W], got {tuple(q_input_i64.shape)}")

    N, C, H, W = q_input_i64.shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    H_out = ((H + 2 * pH - dH * (kH - 1) - 1) // sH) + 1
    W_out = ((W + 2 * pW - dW * (kW - 1) - 1) // sW) + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError(
            f"Invalid output size in extract_patches_int64: H_out={H_out}, W_out={W_out}"
        )

    # Constant zero padding in integer domain
    padded = F.pad(
        q_input_i64,
        (pW, pW, pH, pH),
        mode="constant",
        value=0,
    )

    patch_slices = []

    # Collect one [N, C, H_out, W_out] slice per kernel position
    for kh in range(kH):
        h_start = kh * dH
        h_end = h_start + sH * H_out

        for kw in range(kW):
            w_start = kw * dW
            w_end = w_start + sW * W_out

            patch = padded[:, :, h_start:h_end:sH, w_start:w_end:sW]

            if patch.shape != (N, C, H_out, W_out):
                raise RuntimeError(
                    f"Patch extraction produced wrong shape: got {tuple(patch.shape)}, "
                    f"expected {(N, C, H_out, W_out)}"
                )

            patch_slices.append(patch)

    # Stack kernel positions:
    # [N, C, kH*kW, H_out, W_out]
    patches = torch.stack(patch_slices, dim=2)

    # Reshape to [N, C*kH*kW, L]
    patches = patches.reshape(N, C * kH * kW, H_out * W_out)

    return patches

def conv2d_int_reference(
    q_input: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor],
    input_zero_point: int,
    weight_zero_point: int,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> torch.Tensor:
    """
    Integer Conv2d reference implementation using:
        - manual integer patch extraction
        - integer zero-point correction
        - integer matmul / accumulation
    """
    if q_input.dim() != 4:
        raise ValueError(f"q_input must be 4D [N,C,H,W], got shape {tuple(q_input.shape)}")
    if q_weight.dim() != 4:
        raise ValueError(f"q_weight must be 4D [C_out,C_in/groups,kH,kW], got shape {tuple(q_weight.shape)}")

    N, C_in, H, W = q_input.shape
    C_out, _, kH, kW = q_weight.shape

    if C_in % groups != 0:
        raise ValueError(f"C_in={C_in} must be divisible by groups={groups}")
    if C_out % groups != 0:
        raise ValueError(f"C_out={C_out} must be divisible by groups={groups}")

    C_in_per_group = C_in // groups
    if q_weight.shape[1] != C_in_per_group:
        raise ValueError(
            f"q_weight second dimension must be {C_in_per_group} for "
            f"C_in={C_in}, groups={groups}, got {q_weight.shape[1]}"
        )

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_out = ((H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h) + 1
    W_out = ((W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w) + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError(
            f"Invalid output size: H_out={H_out}, W_out={W_out}. "
            f"Check kernel/stride/padding/dilation."
        )

    q_input_i64 = q_input.to(torch.int64)
    q_weight_i64 = q_weight.to(torch.int64)

    patches = extract_patches_int64(
        q_input_i64=q_input_i64,
        kernel_size=(kH, kW),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dil_h, dil_w),
    )

    L = H_out * W_out
    C_out_per_group = C_out // groups
    K = C_in_per_group * kH * kW

    patches_grouped = patches.view(N, groups, K, L)
    weights_grouped = q_weight_i64.view(groups, C_out_per_group, K)

    if input_zero_point != 0:
        patches_grouped = patches_grouped - int(input_zero_point)
    if weight_zero_point != 0:
        weights_grouped = weights_grouped - int(weight_zero_point)

    outputs = []

    for g in range(groups):
        A = patches_grouped[:, g, :, :].transpose(1, 2).contiguous()   # [N, L, K]
        B = weights_grouped[g].transpose(0, 1).contiguous()            # [K, C_out_per_group]

        out_g = torch.matmul(A, B)                                     # [N, L, C_out_per_group]
        out_g = out_g.transpose(1, 2).contiguous()                     # [N, C_out_per_group, L]
        outputs.append(out_g)

    acc = torch.cat(outputs, dim=1)                                    # [N, C_out, L]

    if q_bias is not None:
        if q_bias.numel() != C_out:
            raise ValueError(f"q_bias must have shape [{C_out}], got {tuple(q_bias.shape)}")
        bias_i64 = q_bias.to(torch.int64).view(1, C_out, 1)
        acc = acc + bias_i64

    acc = acc.view(N, C_out, H_out, W_out)

    acc_min = acc.min().item()
    acc_max = acc.max().item()
    if acc_min < -(2**31) or acc_max > 2**31 - 1:
        raise OverflowError("conv2d_int_reference accumulator exceeded int32 range")

    return acc.to(torch.int32)


def batchnorm2d_int_per_channel(
    q_input: torch.Tensor,
    input_zero_point: int,
    output_zero_point: int,
    multipliers_int: torch.Tensor,
    shifts: torch.Tensor,
    offsets_int: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """
    Vectorized integer BatchNorm2d.

    Runtime form:
        q_y = clamp(
            output_zero_point
            + round_shift_right(multipliers_int[c] * (q_x - input_zero_point), shifts[c])
            + offsets_int[c]
        )
    """
    if q_input.dim() != 4:
        raise ValueError(f"q_input must be 4D [N,C,H,W], got {tuple(q_input.shape)}")

    N, C, H, W = q_input.shape

    multipliers_int = multipliers_int.to(device=q_input.device, dtype=torch.int64).view(
        1, C, 1, 1
    )
    shifts = shifts.to(device=q_input.device, dtype=torch.int64).view(1, C, 1, 1)
    offsets_int = offsets_int.to(device=q_input.device, dtype=torch.int64).view(
        1, C, 1, 1
    )

    x = q_input.to(torch.int64) - int(input_zero_point)
    scaled = x * multipliers_int

    # Need per-channel shift, so handle unique shifts efficiently
    out = torch.zeros_like(scaled)

    unique_shifts = torch.unique(shifts)
    for s in unique_shifts.tolist():
        mask = shifts == s
        out = torch.where(mask, round_shift_right(scaled, int(s)), out)

    out = out + int(output_zero_point) + offsets_int
    out = torch.clamp(out, int(qmin), int(qmax))

    return out.to(torch.int32)


def gelu_lut_int(
    q_input: torch.Tensor,
    lut_values: torch.Tensor,
    input_qmin: int,
) -> torch.Tensor:
    """
    Integer GELU using precomputed LUT.

    LUT convention:
        lut index = q_input - input_qmin

    Example:
        if input_qmin = -128:
            q=-128 -> index 0
            q=127  -> index 255

    Args:
        q_input:
            Integer input tensor
        lut_values:
            1D LUT tensor
        input_qmin:
            Minimum representable input integer

    Returns:
        Integer output tensor
    """
    indices = (q_input.to(torch.int32) - input_qmin).to(torch.long)
    indices = torch.clamp(indices, 0, len(lut_values) - 1)
    output = lut_values[indices]
    return output


def maxpool2d_int(
    q_input: torch.Tensor,
    kernel_size: int | Tuple[int, int],
    stride: Optional[int | Tuple[int, int]] = None,
    padding: int | Tuple[int, int] = 0,
) -> torch.Tensor:
    q_input = q_input.to(torch.int32)
    return F.max_pool2d(
        q_input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def upsample_nearest_int(
    q_input: torch.Tensor,
    scale_factor: int = 2,
) -> torch.Tensor:
    """
    Integer nearest-neighbor upsampling.

    This is just value replication, so it is straightforward in integer arithmetic.
    """
    return q_input.repeat_interleave(scale_factor, dim=2).repeat_interleave(
        scale_factor, dim=3
    )


def concat_int(
    tensors: List[torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    """
    Concatenate integer tensors along the requested dimension.

    Important:
        All tensors should already be in the same quantization domain
        before calling this function.
    """
    return torch.cat(tensors, dim=dim)


def add_int(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """
    Integer add for tensors already aligned to the same quantization domain.

    Important:
        This function assumes:
            - same scale
            - same zero-point
        Alignment/requantization should happen before this op if needed.
    """
    # Add the two integer tensors
    result = q_a.to(torch.int32) + q_b.to(torch.int32)

    # Clamp to [qmin, qmax]
    output = clamp_int_tensor(result, qmin, qmax)

    return output.to(torch.int32)


def apply_conv_op(
    q_input: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor],
    conv_params: Dict[str, Any],
    requant_params: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    rp = requant_params.get("params", requant_params)

    acc_int32 = conv2d_int_reference(
        q_input=q_input,
        q_weight=q_weight,
        q_bias=q_bias,
        input_zero_point=int(conv_params["input_zero_point"]),
        weight_zero_point=int(conv_params["weight_zero_point"]),
        stride=tuple(conv_params["stride"]),
        padding=tuple(conv_params["padding"]),
        dilation=tuple(conv_params["dilation"]),
        groups=int(conv_params["groups"]),
    )

    out_q = requantize_int32(
        acc_int32=acc_int32,
        multiplier_int=int(rp["multiplier_int"]),
        shift=int(rp["shift"]),
        out_zero_point=int(rp["out_zero_point"]),
        qmin=int(rp["qmin"]),
        qmax=int(rp["qmax"]),
    )

    return acc_int32, out_q


def apply_bn_op(
    q_input: torch.Tensor,
    bn_params: Dict[str, Any],
) -> torch.Tensor:
    """
    Execute unfused integer BatchNorm2d op using exported BN params.
    """
    in_qparams = bn_params.get("in_qparams", {})
    input_zero_point = int(in_qparams.get("zero_point", 0))

    out_qparams = bn_params.get("out_qparams", {})
    output_zero_point = int(out_qparams.get("zero_point", 0))
    qmin = int(out_qparams.get("qmin", -128))
    qmax = int(out_qparams.get("qmax", 127))

    multipliers_int = torch.as_tensor(
        bn_params.get("multipliers_int"),
        dtype=torch.int32,
        device=q_input.device,
    )
    shifts = torch.as_tensor(
        bn_params.get("shifts"),
        dtype=torch.int32,
        device=q_input.device,
    )
    offsets_int = torch.as_tensor(
        bn_params.get("offsets_int"),
        dtype=torch.int32,
        device=q_input.device,
    )

    return batchnorm2d_int_per_channel(
        q_input=q_input,
        input_zero_point=input_zero_point,
        output_zero_point=output_zero_point,
        multipliers_int=multipliers_int,
        shifts=shifts,
        offsets_int=offsets_int,
        qmin=qmin,
        qmax=qmax,
    )


def apply_gelu_op(
    q_input: torch.Tensor,
    gelu_params: Dict[str, Any],
) -> torch.Tensor:
    """
    Execute integer GELU LUT op using exported LUT artifact.
    """
    lut_values = torch.as_tensor(
        gelu_params.get("lut_values"),
        dtype=torch.int32,
        device=q_input.device,
    )

    in_qparams = gelu_params.get("in_qparams", {})
    input_qmin = int(in_qparams.get("qmin", -128))

    return gelu_lut_int(
        q_input=q_input,
        lut_values=lut_values,
        input_qmin=input_qmin,
    )
