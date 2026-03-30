"""
quantize.py

Purpose:
    Offline quantization compiler for CMUNeXt PTQ.

Main responsibilities:
    1. Load the prepared float model and calibration statistics.
    2. Compute quantization parameters for weights and activations.
    3. Quantize Conv2d weights to int8.
    4. Quantize biases to int32.
    5. Compute integer requantization parameters (multiplier + shift).
    6. Handle unfused BatchNorm2d layers as separate affine integer ops.
    7. Build GELU LUTs for integer inference.
    8. Export all quantized artifacts for use by int_engine.py and int_ops.py.

Important:
    - This file does NOT run integer inference.
    - This file computes and exports the constants needed for integer inference.
    - This file should operate on the prepared / folded model from prepare_model.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn

from prepare_model import prepare_model_for_ptq, build_module_name_map


@dataclass
class QuantParams:
    """
    Quantization parameters for one tensor domain.

    Fields:
        scale:
            Floating-point scale S
        zero_point:
            Integer zero-point Z
        qmin:
            Minimum representable integer
        qmax:
            Maximum representable integer
        dtype:
            String identifier such as 'int8', 'uint8', 'int32'
        symmetric:
            Whether the quantization scheme is symmetric
    """

    scale: float
    zero_point: int
    qmin: int
    qmax: int
    dtype: str
    symmetric: bool


@dataclass
class QuantizedWeight:
    """
    Quantized Conv2d weight artifact.
    """

    name: str
    qparams: QuantParams
    shape: List[int]
    weight_int: np.ndarray


@dataclass
class QuantizedBias:
    """
    Quantized bias artifact.
    """

    name: str
    scale: float
    zero_point: int
    shape: List[int]
    bias_int: np.ndarray


@dataclass
class RequantParams:
    """
    Integer requantization parameters for mapping int32 accumulator to output integer domain.

    Fields:
        multiplier_int:
            Fixed-point integer multiplier
        shift:
            Right shift amount
        out_zero_point:
            Output tensor zero-point
        qmin, qmax:
            Output clamp range
    """

    multiplier_int: int
    shift: int
    out_zero_point: int
    qmin: int
    qmax: int


@dataclass
class BNIntegerParams:
    """
    Integer parameters for an unfused BatchNorm2d layer.

    Because unfused BN must be treated as a separate affine op:
        y = a_c * x + b_c

    after quantization, we store per-channel integer parameters.

    Fields:
        name:
            Stable module name
        in_qparams:
            Input quantization parameters
        out_qparams:
            Output quantization parameters
        multipliers_int:
            Per-channel fixed-point multipliers
        shifts:
            Per-channel right shifts
        offsets_int:
            Per-channel integer offsets
        num_channels:
            Number of BN channels
    """

    name: str
    in_qparams: Dict[str, Any]
    out_qparams: Dict[str, Any]
    multipliers_int: List[int]
    shifts: List[int]
    offsets_int: List[int]
    num_channels: int


@dataclass
class GELULUT:
    """
    Integer lookup table for GELU.

    Fields:
        name:
            Module name or identifier for this LUT
        in_qparams:
            Input qparams used to interpret LUT indices
        out_qparams:
            Output qparams of the GELU op
        lut_values:
            Integer output values for all possible input codes
    """

    name: str
    in_qparams: Dict[str, Any]
    out_qparams: Dict[str, Any]
    lut_values: List[int]


@dataclass
class ExecutionTraceEntry:
    """
    One executed operation observed during a float forward pass.

    Fields:
        op_name:
            Stable module name from build_module_name_map()
        op_type:
            Module class name, e.g. 'Conv2d', 'GELU', 'BatchNorm2d'
        input_tensor_names:
            Names of tensors consumed by this op
        output_tensor_name:
            Name of tensor produced by this op
        input_shapes:
            Shapes of input tensors
        output_shape:
            Shape of output tensor
        execution_index:
            Order in which this op executed during forward pass
    """

    op_name: str
    op_type: str
    input_tensor_names: List[str]
    output_tensor_name: str
    input_shapes: List[List[int]]
    output_shape: List[int]
    execution_index: int


def get_qrange(dtype: str) -> Tuple[int, int]:
    """
    Return integer range for a quantized dtype.

    Args:
        dtype:
            Supported examples: 'int8', 'uint8', 'int32'

    Returns:
        (qmin, qmax)

    Notes:
        - int8  -> (-128, 127) or sometimes (-127, 127) depending on design
        - uint8 -> (0, 255)
        - int32 -> (-2**31, 2**31 - 1)
    """
    if dtype == "int8":
        return (-128, 127)
    elif dtype == "uint8":
        return (0, 255)
    elif dtype == "int32":
        return (-(2**31), 2**31 - 1)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def choose_qparams_from_minmax(
    min_val: float,
    max_val: float,
    dtype: str = "int8",
    symmetric: bool = False,
) -> QuantParams:
    """
    Compute quantization parameters from observed min/max.

    Args:
        min_val:
            Minimum observed real value
        max_val:
            Maximum observed real value
        dtype:
            Integer dtype for quantization
        symmetric:
            If True, use symmetric quantization around zero.
            If False, use affine quantization.

    Returns:
        QuantParams

    Implementation notes:
        - For symmetric quantization, use abs(max(|min|, |max|))
        - For affine quantization, map [min_val, max_val] into [qmin, qmax]
        - Ensure zero is representable
        - Guard against degenerate min=max cases
    """
    qmin, qmax = get_qrange(dtype)

    # Guard against degenerate case
    if min_val == max_val:
        min_val = min_val - 0.1
        max_val = max_val + 0.1

    if symmetric:
        # Symmetric: use range around zero
        abs_max = max(abs(min_val), abs(max_val))
        scale = abs_max / float(qmax) if abs_max > 0 else 1.0
        zero_point = 0
    else:
        # Affine: map [min_val, max_val] to [qmin, qmax]
        scale = (max_val - min_val) / float(qmax - qmin)
        if scale == 0.0:
            scale = 1.0
        zero_point = int(round(qmin - min_val / scale))
        # Clamp zero_point to valid range
        zero_point = max(qmin, min(qmax, zero_point))

    return QuantParams(
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        dtype=dtype,
        symmetric=symmetric,
    )


def quantize_array(
    x: np.ndarray,
    qparams: QuantParams,
) -> np.ndarray:
    """
    Quantize a floating-point numpy array into the integer domain.

    Formula:
        q = round(x / scale) + zero_point

    then clamp to [qmin, qmax].

    Args:
        x:
            Float numpy array
        qparams:
            Quantization parameters

    Returns:
        Integer numpy array with dtype matching qparams.dtype
    """
    # Apply quantization formula
    q = np.round(x / qparams.scale) + qparams.zero_point

    # Clamp to valid integer range
    q = np.clip(q, qparams.qmin, qparams.qmax)

    # Convert to appropriate integer dtype
    if qparams.dtype == "int8":
        q = q.astype(np.int8)
    elif qparams.dtype == "uint8":
        q = q.astype(np.uint8)
    elif qparams.dtype == "int32":
        q = q.astype(np.int32)
    else:
        raise ValueError(f"Unsupported dtype: {qparams.dtype}")

    return q


def dequantize_array(
    q: np.ndarray,
    qparams: QuantParams,
) -> np.ndarray:
    """
    Dequantize an integer array back to float.

    Formula:
        x = scale * (q - zero_point)

    Useful mainly for debugging / validation.
    """
    # Convert to float to avoid integer overflow
    q_float = q.astype(np.float32)

    # Apply dequantization formula
    x = qparams.scale * (q_float - qparams.zero_point)

    return x


def quantize_conv_weight(
    name: str,
    conv: nn.Conv2d,
    weight_dtype: str = "int8",
    symmetric: bool = True,
) -> QuantizedWeight:
    """
    Quantize one Conv2d weight tensor.

    Args:
        name:
            Stable module name
        conv:
            Conv2d module
        weight_dtype:
            Usually 'int8'
        symmetric:
            Usually True for weights

    Returns:
        QuantizedWeight artifact

    Notes:
        - First implementation can use per-tensor quantization.
        - Later you may upgrade to per-output-channel quantization if needed.
    """
    weight_fp = conv.weight.data.cpu().numpy()

    # Per-tensor quantization: find global min/max
    min_val = float(np.min(weight_fp))
    max_val = float(np.max(weight_fp))

    # Compute quantization parameters
    qparams = choose_qparams_from_minmax(
        min_val=min_val,
        max_val=max_val,
        dtype=weight_dtype,
        symmetric=symmetric,
    )

    # Quantize weights to integer
    weight_int = quantize_array(weight_fp, qparams)

    return QuantizedWeight(
        name=name,
        qparams=qparams,
        shape=list(weight_fp.shape),
        weight_int=weight_int,
    )


def quantize_bias_tensor(
    name: str,
    bias_fp: np.ndarray,
    input_qparams: QuantParams,
    weight_qparams: QuantParams,
) -> QuantizedBias:
    """
    Quantize a bias tensor to int32.

    Standard rule:
        S_bias = S_x * S_w
        Z_bias = 0

    Args:
        name:
            Bias name
        bias_fp:
            Float bias array
        input_qparams:
            Input activation qparams
        weight_qparams:
            Weight qparams

    Returns:
        QuantizedBias artifact
    """
    # Compute bias scale from input and weight scales
    bias_scale = input_qparams.scale * weight_qparams.scale

    # Bias is always zero-pointed at zero
    bias_zero_point = 0

    # Create QuantParams for int32 bias
    bias_qparams = QuantParams(
        scale=bias_scale,
        zero_point=bias_zero_point,
        qmin=-(2**31),
        qmax=2**31 - 1,
        dtype="int32",
        symmetric=False,
    )

    # Quantize bias array to int32
    bias_int = quantize_array(bias_fp, bias_qparams)

    return QuantizedBias(
        name=name,
        scale=bias_scale,
        zero_point=bias_zero_point,
        shape=list(bias_fp.shape),
        bias_int=bias_int,
    )


def quantize_multiplier(real_multiplier: float) -> Tuple[int, int]:
    """
    Convert a floating-point multiplier into fixed-point integer multiplier + shift.

    Returns:
        (multiplier_int, shift)

    Convention:
        real_multiplier ≈ multiplier_int / 2^(31 - shift_adjustment)
        Runtime must match this exactly.
    """
    if real_multiplier == 0.0:
        return 0, 0

    sign = -1 if real_multiplier < 0 else 1
    m = abs(real_multiplier)
    shift = 0

    while m >= 1.0:
        m /= 2.0
        shift += 1

    while m < 0.5:
        m *= 2.0
        shift -= 1

    multiplier_int = int(round(m * (2**31)))
    multiplier_int *= sign

    multiplier_int = max(-(2**31), min(2**31 - 1, multiplier_int))
    return multiplier_int, shift


def build_requant_params(
    input_qparams: QuantParams,
    weight_qparams: QuantParams,
    output_qparams: QuantParams,
) -> RequantParams:
    """
    Build requantization parameters for a Conv2d output path.

    Args:
        input_qparams:
            Input activation qparams
        weight_qparams:
            Weight qparams
        output_qparams:
            Desired output activation qparams

    Returns:
        RequantParams
    """
    # Compute the real multiplier M = (S_x * S_w) / S_y
    real_multiplier = (
        input_qparams.scale * weight_qparams.scale
    ) / output_qparams.scale

    # Convert real multiplier to fixed-point integer representation
    multiplier_int, shift = quantize_multiplier(real_multiplier)

    # Extract output zero-point and range from output qparams
    out_zero_point = output_qparams.zero_point
    qmin = output_qparams.qmin
    qmax = output_qparams.qmax

    return RequantParams(
        multiplier_int=multiplier_int,
        shift=shift,
        out_zero_point=out_zero_point,
        qmin=qmin,
        qmax=qmax,
    )


def extract_bn_affine_params(
    bn: nn.BatchNorm2d,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an inference-time BatchNorm2d layer into per-channel affine parameters.

    In inference mode:
        y_c = gamma_c * (x_c - mean_c) / sqrt(var_c + eps) + beta_c

    which can be rewritten as:
        y_c = a_c * x_c + b_c

    Returns:
        a: np.ndarray of shape [C]
        b: np.ndarray of shape [C]

    These floating-point affine params are later quantized into integer
    multipliers / shifts / offsets.
    """
    # Extract BN parameters in eval mode
    gamma = bn.weight.data.cpu().numpy()  # [C]
    beta = bn.bias.data.cpu().numpy()  # [C]
    mean = bn.running_mean.data.cpu().numpy()  # [C]
    var = bn.running_var.data.cpu().numpy()  # [C]
    eps = bn.eps

    # Compute per-channel affine coefficients
    # y_c = gamma_c * (x_c - mean_c) / sqrt(var_c + eps) + beta_c
    # y_c = (gamma_c / sqrt(var_c + eps)) * x_c + (beta_c - gamma_c * mean_c / sqrt(var_c + eps))

    denom = np.sqrt(var + eps)
    a = gamma / denom  # [C]
    b = beta - (gamma * mean) / denom  # [C]

    return a, b


# Runtime must implement:
# q_y = clamp( Z_y + round_shift( multiplier_int[c] * (q_x - Z_x), shift[c] ) + offsets_int[c] )
def quantize_unfused_batchnorm(
    name: str,
    bn: nn.BatchNorm2d,
    input_qparams: QuantParams,
    output_qparams: QuantParams,
) -> BNIntegerParams:
    """
    Build integer parameters for an unfused BatchNorm2d layer.

    Given:
        x = S_x (q_x - Z_x)
        y = a x + b
        y = S_y (q_y - Z_y)

    derive integer form:
        q_y = Z_y + multiplier_c * (q_x - Z_x) + offset_c

    where multiplier_c is represented using fixed-point integer math.

    Args:
        name:
            Stable BN module name
        bn:
            Unfused BatchNorm2d module
        input_qparams:
            Input qparams to the BN layer
        output_qparams:
            Output qparams from calibration

    Returns:
        BNIntegerParams

    Notes:
        - This should be per-channel.
        - offsets_int must absorb the floating b term and zero-point correction.
        - int_ops.py must implement the exact same interpretation.
    """
    # Extract per-channel affine parameters from BN
    a, b = extract_bn_affine_params(bn)  # a, b are [C]
    num_channels = len(a)

    # Compute per-channel real multipliers and convert to fixed-point
    multipliers_int = []
    shifts = []

    for c in range(num_channels):
        # Real multiplier for channel c: M_c = (a_c * S_x) / S_y
        real_multiplier = (a[c] * input_qparams.scale) / output_qparams.scale
        mult_int, shift = quantize_multiplier(real_multiplier)
        multipliers_int.append(mult_int)
        shifts.append(shift)

    # Compute per-channel integer offsets
    # Floating-point: y_c = a_c * (x - Z_x * S_x / S_x) + b_c
    #               = a_c * x + b_c
    # Integer: q_y = Z_y + multiplier_c * (q_x - Z_x) + offset_c
    #
    # After requantization, we need:
    # offset_c = (b_c / S_y) - (a_c * (-Z_x) * S_x) / S_y
    #          = b_c / S_y + (a_c * Z_x * S_x) / S_y

    offsets_int = []
    for c in range(num_channels):
        # Floating-point offset contribution from bias term
        bias_contribution = b[c] / output_qparams.scale

        # Zero-point correction: a_c * Z_x * S_x / S_y
        zp_contribution = (
            a[c] * input_qparams.zero_point * input_qparams.scale
        ) / output_qparams.scale

        # Total offset
        offset_float = bias_contribution + zp_contribution
        offset_int = int(round(offset_float))
        offsets_int.append(offset_int)

    return BNIntegerParams(
        name=name,
        in_qparams=asdict(input_qparams),
        out_qparams=asdict(output_qparams),
        multipliers_int=multipliers_int,
        shifts=shifts,
        offsets_int=offsets_int,
        num_channels=num_channels,
    )


def gelu_float(x: np.ndarray) -> np.ndarray:
    """
    Reference floating-point GELU matching PyTorch nn.GELU(approximate='none').

    PyTorch default GELU is:
        GELU(x) = x * Phi(x)
                = 0.5 * x * (1 + erf(x / sqrt(2)))

    Args:
        x:
            Input numpy array

    Returns:
        Float numpy array with GELU applied.

    Notes:
        - This should match the model's training-time GELU semantics.
        - Since your trained model used nn.GELU() without approximation,
          this is the correct reference for LUT generation.
    """
    x = x.astype(np.float32, copy=False)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf_vec(x / np.sqrt(2.0)))


def build_gelu_lut(
    name: str,
    input_qparams: QuantParams,
    output_qparams: QuantParams,
) -> GELULUT:
    """
    Build an integer lookup table for GELU.
    """
    # Use the actual quantization range stored in input_qparams
    qmin, qmax = input_qparams.qmin, input_qparams.qmax

    lut_values = []

    for code in range(qmin, qmax + 1):
        # Dequantize integer code to floating-point
        x_float = dequantize_array(np.array([code], dtype=np.int32), input_qparams)[0]

        # Apply floating-point GELU reference
        y_float = gelu_float(np.array([x_float], dtype=np.float32))[0]

        # Quantize GELU output into output domain
        y_quant = quantize_array(np.array([y_float], dtype=np.float32), output_qparams)[
            0
        ]

        lut_values.append(int(y_quant))

    return GELULUT(
        name=name,
        in_qparams=asdict(input_qparams),
        out_qparams=asdict(output_qparams),
        lut_values=lut_values,
    )


def load_calibration_stats(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load activation calibration statistics from calibration_stats.json.

    Args:
        path:
            JSON path

    Returns:
        Dict keyed by module name
    """
    with open(path, "r") as f:
        stats = json.load(f)
    return stats


def build_activation_qparams_from_stats(
    calibration_stats: Dict[str, Dict[str, Any]],
    activation_dtype: str = "int8",
    symmetric: bool = False,
) -> Dict[str, QuantParams]:
    """
    Build activation qparams for all recorded module outputs.

    Args:
        calibration_stats:
            Output from calibrate.py
        activation_dtype:
            Usually 'int8' or 'uint8'
        symmetric:
            Usually False for activations

    Returns:
        Dict[module_name -> QuantParams]

    Notes:
        - This creates output qparams for every calibrated module.
        - During graph assembly, these will be assigned to op outputs.
    """
    activation_qparams = {}

    for module_name, stats in calibration_stats.items():
        # Extract min/max statistics for this module's output
        if "min_val" not in stats or "max_val" not in stats:
            print(f"Warning: module {module_name} missing min/max stats, skipping")
            continue

        min_val = float(stats["min_val"])
        max_val = float(stats["max_val"])

        # Compute quantization parameters from min/max
        qparams = choose_qparams_from_minmax(
            min_val=min_val,
            max_val=max_val,
            dtype=activation_dtype,
            symmetric=symmetric,
        )

        activation_qparams[module_name] = qparams

    return activation_qparams


def should_trace_module(module: nn.Module) -> bool:
    """
    Return True if this module should appear in the runtime execution trace.
    """
    trace_types = (
        nn.Conv2d,
        nn.GELU,
        nn.BatchNorm2d,
        nn.MaxPool2d,
        nn.Upsample,
    )

    if isinstance(module, trace_types):
        return True

    # Trace custom wrapper modules whose outputs feed later modules
    if module.__class__.__name__ == "Residual":
        return True

    return False


def extract_tensor_list(obj: Any) -> List[torch.Tensor]:
    """
    Extract all torch.Tensor objects from a nested hook input/output object.

    Args:
        obj:
            Hook input or output object. Could be:
                - a Tensor
                - a tuple/list of Tensors
                - nested structures

    Returns:
        Flat list of torch.Tensor objects

    Why:
        Forward hooks often receive inputs as tuples.
        This helper makes tensor extraction robust.
    """
    tensors: List[torch.Tensor] = []

    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(extract_tensor_list(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            tensors.extend(extract_tensor_list(item))

    return tensors


@torch.no_grad()
def build_execution_trace(
    model: nn.Module,
    sample_input: torch.Tensor,
    name_map: Dict[int, str],
) -> List[ExecutionTraceEntry]:
    """
    Run one float forward pass and record execution order plus tensor flow
    for relevant modules.

    Args:
        model:
            Prepared / folded float model
        sample_input:
            One representative input tensor, e.g. [1, 3, 256, 256]
        name_map:
            Mapping from id(module) -> stable module name

    Returns:
        Ordered list of ExecutionTraceEntry objects

    High-level idea:
        - Each module output gets a stable tensor name:
              "<module_name>:out"
        - Inputs to each module are matched by Python object identity
          against previously seen tensor outputs.
        - If an input tensor does not match a known produced tensor,
          label it as a source tensor such as "input".

    Important limitations:
        - Functional ops like torch.cat and residual adds are not captured
          automatically by module forward hooks, because they are not modules.
        - This trace still solves most per-module qparam mapping for Conv/GELU/BN/etc.
        - Later, concat/add may need explicit manual handling or model refactoring.
    """
    # This will store the trace entries in execution order.
    trace_entries: List[ExecutionTraceEntry] = []

    # Maps Python object id(tensor) -> logical tensor name.
    # Used to infer which previous op produced a module's input.
    tensor_id_to_name: Dict[int, str] = {}

    # Optional: store tensor shapes for debugging.
    tensor_name_to_shape: Dict[str, List[int]] = {}

    # Counter to record execution order.
    execution_counter = {"value": 0}

    # Hook handles so we can remove them after tracing.
    hook_handles: List[Any] = []

    # Register the sample input as a known source tensor before forward pass.
    # If sample_input is a tensor, map its object id to "input".
    tensor_id_to_name[id(sample_input)] = "input"
    tensor_name_to_shape["input"] = list(sample_input.shape)

    def make_hook(module_name: str, module_type: str):
        """
        Build a forward hook for one module.

        The hook should:
            1. extract tensor inputs
            2. resolve their logical tensor names
            3. assign a new logical name to the output
            4. record an ExecutionTraceEntry
        """

        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            # Extract tensor inputs from hook input tuple.
            input_tensors = extract_tensor_list(inputs)

            # For current design, only handle tensor outputs directly.
            # If output is not a Tensor, skip this op for now.
            if not isinstance(output, torch.Tensor):
                return

            # Resolve input tensor names. If a tensor was produced earlier,
            # use the recorded name; otherwise mark as unknown source.
            input_tensor_names: List[str] = []
            input_shapes: List[List[int]] = []

            for idx, tensor in enumerate(input_tensors):
                tensor_name = tensor_id_to_name.get(
                    id(tensor), f"{module_name}:input{idx}"
                )
                input_tensor_names.append(tensor_name)
                input_shapes.append(list(tensor.shape))

            # Assign output tensor name.
            output_tensor_name = f"{module_name}:out"

            # Record produced tensor mapping for downstream ops.
            tensor_id_to_name[id(output)] = output_tensor_name
            tensor_name_to_shape[output_tensor_name] = list(output.shape)

            # Create trace record.
            entry = ExecutionTraceEntry(
                op_name=module_name,
                op_type=module_type,
                input_tensor_names=input_tensor_names,
                output_tensor_name=output_tensor_name,
                input_shapes=input_shapes,
                output_shape=list(output.shape),
                execution_index=execution_counter["value"],
            )

            trace_entries.append(entry)
            execution_counter["value"] += 1

        return hook

    # Register hooks on relevant modules.
    for module in model.modules():
        if not should_trace_module(module):
            continue

        module_name = name_map.get(id(module))
        if module_name is None:
            continue

        module_type = module.__class__.__name__
        handle = module.register_forward_hook(make_hook(module_name, module_type))
        hook_handles.append(handle)

    # Run one forward pass to populate the trace.
    _ = model(sample_input)

    # Clean up hooks.
    for handle in hook_handles:
        handle.remove()

    # Sort by execution order just to be explicit.
    trace_entries.sort(key=lambda x: x.execution_index)

    return trace_entries


def execution_trace_to_dicts(
    trace_entries: List[ExecutionTraceEntry],
) -> List[Dict[str, Any]]:
    """
    Convert execution trace entries into plain dicts for JSON export or debugging.
    """
    return [asdict(entry) for entry in trace_entries]


def print_execution_trace_summary(
    trace_entries: List[ExecutionTraceEntry],
    max_items: int = 20,
) -> None:
    """
    Print a compact summary of the execution trace.

    Useful for checking whether the runtime order looks reasonable.
    """
    print(f"[trace] Recorded {len(trace_entries)} traced module executions")

    for entry in trace_entries[:max_items]:
        print(
            f"[trace] #{entry.execution_index} "
            f"{entry.op_name} ({entry.op_type}) "
            f"{entry.input_tensor_names} -> {entry.output_tensor_name}"
        )

    if len(trace_entries) > max_items:
        print(f"[trace] ... and {len(trace_entries) - max_items} more entries")


def tensor_name_to_module_name(tensor_name: str) -> Optional[str]:
    """
    Convert a traced tensor name like '<module_name>:out' back to '<module_name>'.

    Returns:
        Module name if tensor_name ends with ':out'
        None for raw input or unknown tensor names
    """
    if tensor_name == "input":
        return None
    if tensor_name.endswith(":out"):
        return tensor_name[:-4]
    return None


def get_output_qparams_for_trace_entry(
    entry: ExecutionTraceEntry,
    activation_qparams: Dict[str, QuantParams],
) -> QuantParams:
    """
    Resolve output qparams for a traced op.

    Assumes activation_qparams are keyed by module/op name.
    """
    if entry.op_name not in activation_qparams:
        raise KeyError(f"Missing output qparams for op: {entry.op_name}")
    return activation_qparams[entry.op_name]


def get_input_qparams_for_trace_entry(
    entry: ExecutionTraceEntry,
    activation_qparams: Dict[str, QuantParams],
    default_input_qparams: QuantParams,
) -> QuantParams:
    """
    Resolve input qparams for a traced op.

    Resolution order:
        1. If input is raw model input -> use default_input_qparams
        2. If input came from '<prev_module>:out' -> use activation_qparams[prev_module]
        3. If input tensor name itself exists in activation_qparams
           (e.g. 'Up_conv5.conv.0:input0') -> use that directly

    This fallback is important for functional producers such as:
        - torch.cat(...)
        - tensor addition
    """
    if len(entry.input_tensor_names) == 0:
        return default_input_qparams

    first_input = entry.input_tensor_names[0]

    if first_input == "input":
        return default_input_qparams

    producer_name = tensor_name_to_module_name(first_input)
    if producer_name is not None and producer_name in activation_qparams:
        return activation_qparams[producer_name]

    # Fallback: direct input calibration key
    if first_input in activation_qparams:
        return activation_qparams[first_input]

    raise KeyError(
        f"Could not resolve input qparams for op '{entry.op_name}' "
        f"from input tensor '{first_input}'"
    )


def collect_quantized_artifacts(
    model: nn.Module,
    name_map: Dict[int, str],
    activation_qparams: Dict[str, QuantParams],
    trace_entries: List[ExecutionTraceEntry],
    default_input_qparams: Optional[QuantParams] = None,
) -> Dict[str, Any]:
    """
    Traverse the prepared model and build quantized artifacts for supported modules.

    Responsibilities:
        - Quantize Conv2d weights
        - Quantize biases
        - Build requant params for Conv2d outputs
        - Build BN integer params for unfused BatchNorm2d
        - Build GELU LUTs
        - Record enough metadata for int_engine.py

    Args:
        model:
            Prepared / folded model
        name_map:
            id(module) -> stable module name
        activation_qparams:
            Output activation qparams from calibration stats, keyed by module name
        trace_entries:
            Runtime execution trace built from one forward pass
        default_input_qparams:
            Qparams for the model input tensor. If None, a default affine int8
            qparams for [0, 1] input are used.

    Returns:
        A nested artifact dict suitable for export
    """
    if default_input_qparams is None:
        # Matches your preprocessing: image.astype(float32) / 255, so input is in [0, 1].
        default_input_qparams = choose_qparams_from_minmax(
            min_val=0.0,
            max_val=1.0,
            dtype="int8",
            symmetric=False,
        )

    artifacts = {
        "model_input_qparams": asdict(default_input_qparams),
        "activation_qparams": {k: asdict(v) for k, v in activation_qparams.items()},
        "quantized_weights": [],
        "quantized_biases": [],
        "requant_params": [],
        "conv_params": [],
        "bn_params": [],
        "gelu_luts": [],
        "execution_trace": [asdict(entry) for entry in trace_entries],
    }

    # Map op/module name -> trace entry
    trace_by_op_name: Dict[str, ExecutionTraceEntry] = {
        entry.op_name: entry for entry in trace_entries
    }

    for module_name, module in model.named_modules():
        stable_name = name_map.get(id(module), module_name)

        # Skip modules that are not part of traced execution
        if stable_name not in trace_by_op_name:
            continue

        entry = trace_by_op_name[stable_name]

        # -------------------------
        # Conv2d
        # -------------------------
        if isinstance(module, nn.Conv2d):
            output_qparams = get_output_qparams_for_trace_entry(
                entry, activation_qparams
            )

            input_qparams = get_input_qparams_for_trace_entry(
                entry=entry,
                activation_qparams=activation_qparams,
                default_input_qparams=default_input_qparams,
            )

            quantized_weight = quantize_conv_weight(
                name=stable_name,
                conv=module,
                weight_dtype="int8",
                symmetric=True,
            )
            artifacts["quantized_weights"].append(asdict(quantized_weight))

            if module.bias is not None:
                bias_fp = module.bias.detach().cpu().numpy()
                quantized_bias = quantize_bias_tensor(
                    name=stable_name,
                    bias_fp=bias_fp,
                    input_qparams=input_qparams,
                    weight_qparams=quantized_weight.qparams,
                )
                artifacts["quantized_biases"].append(asdict(quantized_bias))

            requant = build_requant_params(
                input_qparams=input_qparams,
                weight_qparams=quantized_weight.qparams,
                output_qparams=output_qparams,
            )

            artifacts["requant_params"].append(
                {
                    "name": stable_name,
                    "params": asdict(requant),
                    "input_tensor_names": entry.input_tensor_names,
                    "output_tensor_name": entry.output_tensor_name,
                }
            )

            artifacts["conv_params"].append(
                {
                    "name": stable_name,
                    "input_tensor_names": entry.input_tensor_names,
                    "output_tensor_name": entry.output_tensor_name,
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                    "dilation": list(module.dilation),
                    "groups": int(module.groups),
                    "input_zero_point": int(input_qparams.zero_point),
                    "weight_zero_point": int(quantized_weight.qparams.zero_point),
                }
            )

        # -------------------------
        # Unfused BatchNorm2d
        # -------------------------
        elif isinstance(module, nn.BatchNorm2d):
            output_qparams = get_output_qparams_for_trace_entry(
                entry, activation_qparams
            )

            input_qparams = get_input_qparams_for_trace_entry(
                entry=entry,
                activation_qparams=activation_qparams,
                default_input_qparams=default_input_qparams,
            )

            bn_params = quantize_unfused_batchnorm(
                name=stable_name,
                bn=module,
                input_qparams=input_qparams,
                output_qparams=output_qparams,
            )
            artifacts["bn_params"].append(
                {
                    **asdict(bn_params),
                    "input_tensor_names": entry.input_tensor_names,
                    "output_tensor_name": entry.output_tensor_name,
                }
            )

        # -------------------------
        # GELU
        # -------------------------
        elif isinstance(module, nn.GELU):
            output_qparams = get_output_qparams_for_trace_entry(
                entry, activation_qparams
            )

            input_qparams = get_input_qparams_for_trace_entry(
                entry=entry,
                activation_qparams=activation_qparams,
                default_input_qparams=default_input_qparams,
            )

            gelu_lut = build_gelu_lut(
                name=stable_name,
                input_qparams=input_qparams,
                output_qparams=output_qparams,
            )
            artifacts["gelu_luts"].append(
                {
                    **asdict(gelu_lut),
                    "input_tensor_names": entry.input_tensor_names,
                    "output_tensor_name": entry.output_tensor_name,
                }
            )

    return artifacts


def make_jsonable(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable structures.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return make_jsonable(asdict(obj))
    elif isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_jsonable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_quantized_artifacts(
    artifacts: Dict[str, Any],
    export_dir: str,
) -> None:
    """
    Save quantized artifacts to disk.

    Output layout:
        export_dir/
            weights.npz
            biases.npz
            qparams.json
            requant_params.json
            bn_params.json
            gelu_luts.json
            execution_trace.json
            metadata.json

    Notes:
        - Large integer arrays such as weights/biases are stored in NPZ
        - Control/config data is stored in JSON
        - This function avoids mutating the original artifacts dict
    """
    os.makedirs(export_dir, exist_ok=True)

    # Convert all artifacts into JSON/NumPy-friendly form
    artifacts_jsonable = make_jsonable(artifacts)

    quantized_weights = artifacts_jsonable.get("quantized_weights", [])
    quantized_biases = artifacts_jsonable.get("quantized_biases", [])
    requant_params = artifacts_jsonable.get("requant_params", [])
    bn_params = artifacts_jsonable.get("bn_params", [])
    gelu_luts = artifacts_jsonable.get("gelu_luts", [])
    execution_trace = artifacts_jsonable.get("execution_trace", [])
    conv_params = artifacts_jsonable.get("conv_params", [])
    activation_qparams = artifacts_jsonable.get("activation_qparams", {})
    # ----------------------------
    # Save large arrays separately
    # ----------------------------
    weights_data: Dict[str, np.ndarray] = {}
    biases_data: Dict[str, np.ndarray] = {}

    # Make metadata copies so we do not mutate the original jsonable artifacts
    weight_metadata: List[Dict[str, Any]] = []
    bias_metadata: List[Dict[str, Any]] = []

    for weight_info in quantized_weights:
        weight_info_copy = dict(weight_info)
        weight_name = weight_info_copy["name"]

        if "weight_int" not in weight_info_copy:
            raise KeyError(f"Missing 'weight_int' for quantized weight: {weight_name}")

        weights_data[weight_name] = np.array(weight_info_copy["weight_int"])
        weight_info_copy.pop("weight_int")
        weight_metadata.append(weight_info_copy)

    for bias_info in quantized_biases:
        bias_info_copy = dict(bias_info)
        bias_name = bias_info_copy["name"]

        if "bias_int" not in bias_info_copy:
            raise KeyError(f"Missing 'bias_int' for quantized bias: {bias_name}")

        biases_data[bias_name] = np.array(bias_info_copy["bias_int"])
        bias_info_copy.pop("bias_int")
        bias_metadata.append(bias_info_copy)

    if weights_data:
        np.savez_compressed(
            os.path.join(export_dir, "weights.npz"),
            **weights_data,
        )

    if biases_data:
        np.savez_compressed(
            os.path.join(export_dir, "biases.npz"),
            **biases_data,
        )

    # ----------------------------
    # Save structured JSON files
    # ----------------------------

    # 1. qparams summary extracted from weight/bias metadata + BN + GELU artifacts
    qparams_summary = {
        "model_input_qparams": artifacts_jsonable.get("model_input_qparams", None),
        "quantized_weights": weight_metadata,
        "quantized_biases": bias_metadata,
    }
    with open(os.path.join(export_dir, "qparams.json"), "w") as f:
        json.dump(qparams_summary, f, indent=2, sort_keys=True)

    # 2. Requant params
    with open(os.path.join(export_dir, "requant_params.json"), "w") as f:
        json.dump(requant_params, f, indent=2, sort_keys=True)

    # 3. BN params
    with open(os.path.join(export_dir, "bn_params.json"), "w") as f:
        json.dump(bn_params, f, indent=2, sort_keys=True)

    # 4. GELU LUTs
    with open(os.path.join(export_dir, "gelu_luts.json"), "w") as f:
        json.dump(gelu_luts, f, indent=2, sort_keys=True)

    # 5. Execution trace
    with open(os.path.join(export_dir, "execution_trace.json"), "w") as f:
        json.dump(execution_trace, f, indent=2, sort_keys=True)

    with open(os.path.join(export_dir, "conv_params.json"), "w") as f:
        json.dump(conv_params, f, indent=2, sort_keys=True)

    with open(os.path.join(export_dir, "activation_qparams.json"), "w") as f:
        json.dump(activation_qparams, f, indent=2, sort_keys=True)
    # 6. High-level metadata / index
    metadata = {
        "num_quantized_weights": len(weight_metadata),
        "num_quantized_biases": len(bias_metadata),
        "num_activation_qparams": len(activation_qparams),
        "num_requant_params": len(requant_params),
        "num_conv_params": len(conv_params),
        "num_bn_params": len(bn_params),
        "num_gelu_luts": len(gelu_luts),
        "num_trace_entries": len(execution_trace),
        "files": {
            "weights": "weights.npz" if weights_data else None,
            "biases": "biases.npz" if biases_data else None,
            "activation_qparams": "activation_qparams.json",
            "qparams": "qparams.json",
            "requant_params": "requant_params.json",
            "conv_params": "conv_params.json",
            "bn_params": "bn_params.json",
            "gelu_luts": "gelu_luts.json",
            "execution_trace": "execution_trace.json",
            "decomposed_execution_trace": None,
        },
    }

    with open(os.path.join(export_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def run_quantization(
    checkpoint_path: str,
    calibration_stats_path: str,
    variant: str,
    export_dir: str,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end offline quantization pipeline.

    Steps:
        1. Prepare the model using prepare_model_for_ptq()
        2. Build stable module name map
        3. Build runtime execution trace
        4. Load calibration stats
        5. Build activation qparams
        6. Traverse model and build quantized artifacts
        7. Save artifacts to disk

    Returns:
        Artifacts dict
    """
    # Step 1: Prepare the model for PTQ
    _, model, _ = prepare_model_for_ptq(
        checkpoint_path=checkpoint_path,
        variant=variant,
        map_location=map_location,
        strict=strict,
    )

    # Step 2: Build stable module name map
    name_map = build_module_name_map(model)

    # Step 3: Build execution trace using one sample input
    # This is only to recover actual module execution order and tensor flow.
    sample_input = torch.randn(1, 3, 256, 256, device=map_location)
    trace_entries = build_execution_trace(model, sample_input, name_map)
    print_execution_trace_summary(trace_entries, max_items=20)

    # Step 4: Load calibration statistics
    calibration_stats = load_calibration_stats(calibration_stats_path)

    # Step 5: Build activation quantization parameters from calibration stats
    activation_qparams = build_activation_qparams_from_stats(
        calibration_stats,
        activation_dtype="int8",
        symmetric=False,
    )

    # Default model-input qparams based on your actual preprocessing:
    # image.astype(float32) / 255, so the raw model input is in [0, 1].
    default_input_qparams = choose_qparams_from_minmax(
        min_val=0.0,
        max_val=1.0,
        dtype="int8",
        symmetric=False,
    )

    # Step 6: Traverse model and collect quantized artifacts
    artifacts = collect_quantized_artifacts(
        model=model,
        name_map=name_map,
        activation_qparams=activation_qparams,
        trace_entries=trace_entries,
        default_input_qparams=default_input_qparams,
    )

    # Step 7: Save artifacts to disk
    save_quantized_artifacts(
        artifacts=artifacts,
        export_dir=export_dir,
    )

    print(f"[quantize] Quantization complete. Artifacts saved to {export_dir}")
    print(f"[quantize] Quantized weights: {len(artifacts['quantized_weights'])}")
    print(f"[quantize] Quantized biases: {len(artifacts['quantized_biases'])}")
    print(f"[quantize] Requant params: {len(artifacts['requant_params'])}")
    print(f"[quantize] BN params: {len(artifacts['bn_params'])}")
    print(f"[quantize] GELU LUTs: {len(artifacts['gelu_luts'])}")

    return artifacts


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args for quantization script.
    """
    parser = argparse.ArgumentParser(
        description="Offline PTQ artifact generation for CMUNeXt"
    )

    parser.add_argument(
        "checkpoint_path", type=str, help="Path to trained CMUNeXt .pth checkpoint"
    )
    parser.add_argument(
        "--calibration-stats-path",
        type=str,
        default="calibration_stats.json",
        help="Path to calibration stats JSON produced by calibrate.py",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "small", "large"],
        help="CMUNeXt model variant",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="quantized_artifacts",
        help="Directory to save quantized artifacts",
    )
    parser.add_argument(
        "--map-location",
        type=str,
        default="cpu",
        help="Device mapping for checkpoint loading",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Use strict checkpoint loading"
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point for offline PTQ artifact generation.
    """
    args = parse_args()

    print(f"[quantize] Loading checkpoint: {args.checkpoint_path}")
    print(f"[quantize] Loading calibration stats: {args.calibration_stats_path}")
    print(f"[quantize] Export directory: {args.export_dir}")

    artifacts = run_quantization(
        checkpoint_path=args.checkpoint_path,
        calibration_stats_path=args.calibration_stats_path,
        variant=args.variant,
        export_dir=args.export_dir,
        map_location=args.map_location,
        strict=args.strict,
    )

    print("[quantize] Quantization completed successfully.")
    print(f"[quantize] Saved artifacts to: {args.export_dir}")
    print(
        f"[quantize] Quantized weights: {len(artifacts.get('quantized_weights', []))}"
    )
    print(f"[quantize] Quantized biases: {len(artifacts.get('quantized_biases', []))}")
    print(f"[quantize] BN params: {len(artifacts.get('bn_params', []))}")
    print(f"[quantize] GELU LUTs: {len(artifacts.get('gelu_luts', []))}")


if __name__ == "__main__":
    main()
