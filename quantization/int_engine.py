"""
int_engine.py

Purpose:
    Execute quantized CMUNeXt inference using exported PTQ artifacts and
    integer kernels from int_ops.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from int_ops import (
    apply_conv_op,
    apply_bn_op,
    apply_gelu_op,
    maxpool2d_int,
    upsample_nearest_int,
    concat_int,
    add_int,
)

@dataclass
class EngineTensor:
    """
    One runtime tensor stored in the engine.

    Fields:
        name:
            Stable tensor identifier
        data:
            Integer tensor
        qparams:
            Dict containing scale / zero_point / qmin / qmax / dtype
        producer:
            Name of the op that produced this tensor
    """
    name: str
    data: torch.Tensor
    qparams: Dict[str, Any]
    producer: Optional[str] = None


@dataclass
class EngineOp:
    """
    One operation in the runtime execution plan.

    Fields:
        name:
            Stable op name
        op_type:
            Examples: 'Conv2d', 'BatchNorm2d', 'GELU', 'MaxPool2d',
                      'Upsample', 'Concat', 'Add'
        input_names:
            Names of input tensors
        output_name:
            Name of output tensor
        params:
            Exported quantized parameters / metadata for the op
    """
    name: str
    op_type: str
    input_names: List[str]
    output_name: str
    params: Dict[str, Any]

def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load a .npz file as a dict of numpy arrays.
    """
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}

def load_quantized_artifacts(export_dir: str) -> Dict[str, Any]:
    """
    Load all exported quantized artifacts.

    Preferred execution plan source:
        - decomposed_execution_trace.json
    Fallback:
        - execution_trace.json
    """
    artifacts: Dict[str, Any] = {}

    weights_path = os.path.join(export_dir, "weights.npz")
    biases_path = os.path.join(export_dir, "biases.npz")
    qparams_path = os.path.join(export_dir, "qparams.json")
    activation_qparams_path = os.path.join(export_dir, "activation_qparams.json")
    requant_path = os.path.join(export_dir, "requant_params.json")
    conv_params_path = os.path.join(export_dir, "conv_params.json")
    bn_params_path = os.path.join(export_dir, "bn_params.json")
    gelu_luts_path = os.path.join(export_dir, "gelu_luts.json")
    trace_path = os.path.join(export_dir, "execution_trace.json")
    decomposed_trace_path = os.path.join(export_dir, "decomposed_execution_trace.json")
    metadata_path = os.path.join(export_dir, "metadata.json")

    artifacts["weights"] = load_npz(weights_path) if os.path.exists(weights_path) else {}
    artifacts["biases"] = load_npz(biases_path) if os.path.exists(biases_path) else {}

    artifacts["qparams"] = load_json(qparams_path) if os.path.exists(qparams_path) else {}
    artifacts["activation_qparams"] = (
        load_json(activation_qparams_path) if os.path.exists(activation_qparams_path) else {}
    )
    artifacts["requant_params"] = load_json(requant_path) if os.path.exists(requant_path) else []
    artifacts["conv_params"] = load_json(conv_params_path) if os.path.exists(conv_params_path) else []
    artifacts["bn_params"] = load_json(bn_params_path) if os.path.exists(bn_params_path) else []
    artifacts["gelu_luts"] = load_json(gelu_luts_path) if os.path.exists(gelu_luts_path) else []
    artifacts["execution_trace"] = load_json(trace_path) if os.path.exists(trace_path) else []
    artifacts["decomposed_execution_trace"] = (
        load_json(decomposed_trace_path) if os.path.exists(decomposed_trace_path) else []
    )
    artifacts["metadata"] = load_json(metadata_path) if os.path.exists(metadata_path) else {}

    return artifacts

def index_artifacts_by_name(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping:
        item["name"] -> item
    """
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in items:
        name = item.get("name")
        if name is None:
            continue
        indexed[name] = item
    return indexed

def get_weight_tensor(
    artifacts: Dict[str, Any],
    op_name: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Fetch quantized weight tensor from weights.npz.
    """
    weights = artifacts.get("weights", {})
    if op_name not in weights:
        raise KeyError(f"Missing quantized weight for op: {op_name}")

    arr = weights[op_name]
    return torch.as_tensor(arr, dtype=torch.int32, device=device)


def get_bias_tensor(
    artifacts: Dict[str, Any],
    op_name: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Fetch quantized bias tensor from biases.npz if present.
    """
    biases = artifacts.get("biases", {})
    if op_name not in biases:
        return None

    arr = biases[op_name]
    return torch.as_tensor(arr, dtype=torch.int32, device=device)

def quantize_input_tensor(
    x_fp: torch.Tensor,
    input_qparams: Dict[str, Any],
) -> torch.Tensor:
    """
    Quantize float input tensor into integer domain.
    """
    scale = float(input_qparams["scale"])
    zero_point = int(input_qparams["zero_point"])
    qmin = int(input_qparams["qmin"])
    qmax = int(input_qparams["qmax"])

    q = torch.round(x_fp / scale).to(torch.int32) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int32)


def dequantize_output_tensor(
    q: torch.Tensor,
    qparams: Dict[str, Any],
) -> torch.Tensor:
    """
    Dequantize integer tensor back to float.
    """
    scale = float(qparams["scale"])
    zero_point = int(qparams["zero_point"])
    return scale * (q.to(torch.float32) - zero_point)

def create_tensor_store() -> Dict[str, EngineTensor]:
    """
    Create runtime tensor store.
    """
    return {}


def add_tensor_to_store(
    store: Dict[str, EngineTensor],
    name: str,
    data: torch.Tensor,
    qparams: Dict[str, Any],
    producer: Optional[str] = None,
) -> None:
    """
    Insert one tensor into runtime store.
    """
    store[name] = EngineTensor(
        name=name,
        data=data,
        qparams=qparams,
        producer=producer,
    )


def get_tensor_from_store(
    store: Dict[str, EngineTensor],
    name: str,
) -> EngineTensor:
    """
    Fetch one tensor from runtime store.
    """
    if name not in store:
        raise KeyError(f"Tensor '{name}' not found in tensor store")
    return store[name]

def load_engine_plan(artifacts: Dict[str, Any]) -> List[EngineOp]:
    """
    Build runtime execution plan.

    Preference:
        1. decomposed_execution_trace.json
        2. execution_trace.json
    """
    trace_entries = artifacts.get("decomposed_execution_trace", [])
    if not trace_entries:
        trace_entries = artifacts.get("execution_trace", [])

    plan: List[EngineOp] = []

    for entry in trace_entries:
        plan.append(
            EngineOp(
                name=entry["op_name"],
                op_type=entry["op_type"],
                input_names=entry["input_tensor_names"],
                output_name=entry["output_tensor_name"],
                params={},
            )
        )
    return plan

def execute_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    artifacts: Dict[str, Any],
    conv_params_by_name: Dict[str, Dict[str, Any]],
    requant_params_by_name: Dict[str, Dict[str, Any]],
    bn_params_by_name: Dict[str, Dict[str, Any]],
    gelu_params_by_name: Dict[str, Dict[str, Any]],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Dispatch one EngineOp.
    """
    if op.op_type == "Conv2d":
        execute_conv_op(
            op=op,
            tensor_store=tensor_store,
            artifacts=artifacts,
            conv_params_by_name=conv_params_by_name,
            requant_params_by_name=requant_params_by_name,
            output_qparams_map=output_qparams_map,
            trace_store=trace_store,
        )
    elif op.op_type == "BatchNorm2d":
        execute_bn_op(
            op=op,
            tensor_store=tensor_store,
            bn_params_by_name=bn_params_by_name,
            output_qparams_map=output_qparams_map,
            trace_store=trace_store,
        )
    elif op.op_type == "GELU":
        execute_gelu_op(
            op=op,
            tensor_store=tensor_store,
            gelu_params_by_name=gelu_params_by_name,
            output_qparams_map=output_qparams_map,
            trace_store=trace_store,
        )
    elif op.op_type == "MaxPool2d":
        execute_maxpool_op(
            op=op,
            tensor_store=tensor_store,
            output_qparams_map=output_qparams_map,
            trace_store=trace_store,
        )
    elif op.op_type == "Upsample":
        execute_upsample_op(
            op=op,
            tensor_store=tensor_store,
            output_qparams_map=output_qparams_map,
            trace_store=trace_store,
        )
    elif op.op_type == "Concat":
        execute_concat_op(
            op=op,
            tensor_store=tensor_store,
            output_qparams_map=output_qparams_map,
            trace_store=trace_store,
        )
    elif op.op_type == "Add":
        execute_add_op(
            op=op,
            tensor_store=tensor_store,
            output_qparams_map=output_qparams_map,
            trace_store=trace_store,
        )
    else:
        raise NotImplementedError(f"Unsupported op type in engine: {op.op_type}")

def execute_conv_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    artifacts: Dict[str, Any],
    conv_params_by_name: Dict[str, Dict[str, Any]],
    requant_params_by_name: Dict[str, Dict[str, Any]],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute one quantized Conv2d op.
    """
    input_tensor = get_tensor_from_store(tensor_store, op.input_names[0])

    conv_params = conv_params_by_name[op.name]
    requant_params = requant_params_by_name[op.name]

    q_weight = get_weight_tensor(artifacts, op.name, input_tensor.data.device)
    q_bias = get_bias_tensor(artifacts, op.name, input_tensor.data.device)

    acc_int32, out_q = apply_conv_op(
        q_input=input_tensor.data,
        q_weight=q_weight,
        q_bias=q_bias,
        conv_params=conv_params,
        requant_params=requant_params,
    )

    out_qparams = output_qparams_map.get(op.output_name, {})

    add_tensor_to_store(
        tensor_store,
        name=op.output_name,
        data=out_q,
        qparams=out_qparams,
        producer=op.name,
    )

    if trace_store is not None:
        record_trace_entry(
            trace_store,
            op.name,
            {
                "op_type": op.op_type,
                "input_names": op.input_names,
                "output_name": op.output_name,
                "accumulator_int32": acc_int32.detach().cpu(),
                "output_q": out_q.detach().cpu(),
            },
        )

def execute_bn_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    bn_params_by_name: Dict[str, Dict[str, Any]],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute one integer BN op.
    """
    input_tensor = get_tensor_from_store(tensor_store, op.input_names[0])
    bn_params = bn_params_by_name[op.name]

    out_q = apply_bn_op(
        q_input=input_tensor.data,
        bn_params=bn_params,
    )

    out_qparams = output_qparams_map.get(op.output_name, bn_params["out_qparams"])

    add_tensor_to_store(
        tensor_store,
        name=op.output_name,
        data=out_q,
        qparams=out_qparams,
        producer=op.name,
    )

    if trace_store is not None:
        record_trace_entry(
            trace_store,
            op.name,
            {
                "op_type": op.op_type,
                "input_names": op.input_names,
                "output_name": op.output_name,
                "output_q": out_q.detach().cpu(),
            },
        )

def execute_gelu_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    gelu_params_by_name: Dict[str, Dict[str, Any]],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute one GELU LUT op.
    """
    input_tensor = get_tensor_from_store(tensor_store, op.input_names[0])
    gelu_params = gelu_params_by_name[op.name]

    out_q = apply_gelu_op(
        q_input=input_tensor.data,
        gelu_params=gelu_params,
    )

    out_qparams = output_qparams_map.get(op.output_name, gelu_params["out_qparams"])

    add_tensor_to_store(
        tensor_store,
        name=op.output_name,
        data=out_q,
        qparams=out_qparams,
        producer=op.name,
    )

    if trace_store is not None:
        record_trace_entry(
            trace_store,
            op.name,
            {
                "op_type": op.op_type,
                "input_names": op.input_names,
                "output_name": op.output_name,
                "output_q": out_q.detach().cpu(),
            },
        )

def execute_maxpool_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute integer max-pool op.

    Current limitation:
        kernel/stride/padding are not yet exported explicitly, so this
        assumes the CMUNeXt Maxpool configuration is fixed.
    """
    input_tensor = get_tensor_from_store(tensor_store, op.input_names[0])

    out_q = maxpool2d_int(
        q_input=input_tensor.data,
        kernel_size=2,
        stride=2,
        padding=0,
    )

    out_qparams = output_qparams_map.get(op.output_name, input_tensor.qparams)

    add_tensor_to_store(
        tensor_store,
        name=op.output_name,
        data=out_q,
        qparams=out_qparams,
        producer=op.name,
    )

    if trace_store is not None:
        record_trace_entry(
            trace_store,
            op.name,
            {
                "op_type": op.op_type,
                "input_names": op.input_names,
                "output_name": op.output_name,
                "output_q": out_q.detach().cpu(),
            },
        )

def execute_upsample_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute nearest-neighbor integer upsample.

    Current limitation:
        assumes scale_factor=2 to match your retrained CMUNeXt.
    """
    input_tensor = get_tensor_from_store(tensor_store, op.input_names[0])

    out_q = upsample_nearest_int(
        q_input=input_tensor.data,
        scale_factor=2,
    )

    out_qparams = output_qparams_map.get(op.output_name, input_tensor.qparams)

    add_tensor_to_store(
        tensor_store,
        name=op.output_name,
        data=out_q,
        qparams=out_qparams,
        producer=op.name,
    )

    if trace_store is not None:
        record_trace_entry(
            trace_store,
            op.name,
            {
                "op_type": op.op_type,
                "input_names": op.input_names,
                "output_name": op.output_name,
                "output_q": out_q.detach().cpu(),
            },
        )

def execute_residual_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute one residual op.

    Important limitation:
        Current execution_trace.json records Residual as an op, but does not
        yet explicitly encode both add operands. So this function cannot be
        fully correct until the execution plan is enriched.

    Temporary behavior:
        pass through the first input tensor unchanged.

    This is only a placeholder and must be replaced by real residual-add logic.
    """
    input_tensor = get_tensor_from_store(tensor_store, op.input_names[0])

    out_q = input_tensor.data.clone()
    out_qparams = output_qparams_map.get(op.output_name, input_tensor.qparams)

    add_tensor_to_store(
        tensor_store,
        name=op.output_name,
        data=out_q,
        qparams=out_qparams,
        producer=op.name,
    )

    if trace_store is not None:
        record_trace_entry(
            trace_store,
            op.name,
            {
                "op_type": op.op_type,
                "input_names": op.input_names,
                "output_name": op.output_name,
                "output_q": out_q.detach().cpu(),
                "warning": "Residual op currently uses placeholder pass-through behavior.",
            },
        )

def execute_concat_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute one integer concat op.
    """
    input_tensors = [get_tensor_from_store(tensor_store, name) for name in op.input_names]

    out_q = concat_int(
        tensors=[t.data for t in input_tensors],
        dim=1,
    )

    out_qparams = output_qparams_map.get(op.output_name, input_tensors[0].qparams)

    add_tensor_to_store(
        tensor_store,
        name=op.output_name,
        data=out_q,
        qparams=out_qparams,
        producer=op.name,
    )

    if trace_store is not None:
        record_trace_entry(
            trace_store,
            op.name,
            {
                "op_type": op.op_type,
                "input_names": op.input_names,
                "output_name": op.output_name,
                "output_q": out_q.detach().cpu(),
            },
        )

def execute_add_op(
    op: EngineOp,
    tensor_store: Dict[str, EngineTensor],
    output_qparams_map: Dict[str, Dict[str, Any]],
    trace_store: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute one integer add op.

    Assumes both inputs are already in the same quantization domain.
    """
    a = get_tensor_from_store(tensor_store, op.input_names[0])
    b = get_tensor_from_store(tensor_store, op.input_names[1])

    out_qparams = output_qparams_map.get(op.output_name, a.qparams)

    qmin = int(out_qparams.get("qmin", -128))
    qmax = int(out_qparams.get("qmax", 127))

    out_q = add_int(
        q_a=a.data,
        q_b=b.data,
        qmin=qmin,
        qmax=qmax,
    )

    add_tensor_to_store(
        tensor_store,
        name=op.output_name,
        data=out_q,
        qparams=out_qparams,
        producer=op.name,
    )

    if trace_store is not None:
        record_trace_entry(
            trace_store,
            op.name,
            {
                "op_type": op.op_type,
                "input_names": op.input_names,
                "output_name": op.output_name,
                "output_q": out_q.detach().cpu(),
            },
        )

def create_trace_store() -> Dict[str, Any]:
    """
    Create execution trace store for intermediate runtime values.
    """
    return {}

def record_trace_entry(
    trace_store: Dict[str, Any],
    op_name: str,
    payload: Dict[str, Any],
) -> None:
    """
    Record one op execution entry.
    """
    trace_store[op_name] = payload

def build_output_qparams_map(artifacts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping:
        output_tensor_name -> full qparams

    Uses activation_qparams.json exported by quantize.py.
    """
    output_qparams_map: Dict[str, Dict[str, Any]] = {}

    activation_qparams = artifacts.get("activation_qparams", {})

    # Conv outputs
    for item in artifacts.get("requant_params", []):
        op_name = item["name"]
        out_tensor = item["output_tensor_name"]
        if op_name in activation_qparams:
            output_qparams_map[out_tensor] = activation_qparams[op_name]

    # BN outputs
    for item in artifacts.get("bn_params", []):
        op_name = item["name"]
        out_tensor = item["output_tensor_name"]
        if op_name in activation_qparams:
            output_qparams_map[out_tensor] = activation_qparams[op_name]
        else:
            output_qparams_map[out_tensor] = item["out_qparams"]

    # GELU outputs
    for item in artifacts.get("gelu_luts", []):
        op_name = item["name"]
        out_tensor = item["output_tensor_name"]
        if op_name in activation_qparams:
            output_qparams_map[out_tensor] = activation_qparams[op_name]
        else:
            output_qparams_map[out_tensor] = item["out_qparams"]

    return output_qparams_map

def run_integer_inference(
    x_fp: torch.Tensor,
    artifacts: Dict[str, Any],
    return_trace: bool = False,
) -> Dict[str, Any]:
    """
    Run full integer inference on one input tensor.

    Notes:
        - This currently supports module ops traced in execution_trace.json.
        - Functional concat/add structure is not fully encoded yet.
        - Residual op is currently placeholder-only.
    """
    plan = load_engine_plan(artifacts)

    conv_params_by_name = index_artifacts_by_name(artifacts.get("conv_params", []))
    requant_params_by_name = index_artifacts_by_name(artifacts.get("requant_params", []))
    bn_params_by_name = index_artifacts_by_name(artifacts.get("bn_params", []))
    gelu_params_by_name = index_artifacts_by_name(artifacts.get("gelu_luts", []))

    output_qparams_map = build_output_qparams_map(artifacts)

    qparams_json = artifacts.get("qparams", {})
    input_qparams = qparams_json.get("model_input_qparams", None)
    if input_qparams is None:
        raise KeyError("Missing model_input_qparams in qparams.json")

    q_input = quantize_input_tensor(x_fp, input_qparams)

    tensor_store = create_tensor_store()
    add_tensor_to_store(
        tensor_store,
        name="input",
        data=q_input,
        qparams=input_qparams,
        producer=None,
    )

    trace_store = create_trace_store() if return_trace else None

    for op in plan:
        execute_op(
            op=op,
            tensor_store=tensor_store,
            artifacts=artifacts,
            conv_params_by_name=conv_params_by_name,
            requant_params_by_name=requant_params_by_name,
            bn_params_by_name=bn_params_by_name,
            gelu_params_by_name=gelu_params_by_name,
            output_qparams_map=output_qparams_map,
            trace_store=trace_store,
        )

    if len(plan) == 0:
        raise RuntimeError("Execution plan is empty")

    final_output_name = plan[-1].output_name
    final_tensor = get_tensor_from_store(tensor_store, final_output_name)

    result = {
        "output_int": final_tensor.data,
        "output_qparams": final_tensor.qparams,
    }

    if "scale" in final_tensor.qparams and final_tensor.qparams.get("scale") is not None:
        result["output_fp"] = dequantize_output_tensor(final_tensor.data, final_tensor.qparams)

    if return_trace:
        result["trace"] = trace_store

    return result

def load_input_image_as_tensor(
    image_path: str,
    image_size: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    Load one input image using the same preprocessing as calibration:
        - cv2.imread
        - no BGR->RGB conversion
        - resize only
        - /255
        - HWC -> CHW
        - add batch dimension
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return torch.from_numpy(image).to(torch.float32)

def parse_args() -> Any:
    """
    Parse CLI args for integer-engine smoke testing.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Integer CMUNeXt engine smoke test")
    parser.add_argument("--artifacts-dir", type=str, required=True, help="Directory with quantized artifacts")
    parser.add_argument("--image-sample", type=str, required=True, help="Path to one test image")
    parser.add_argument("--return-trace", action="store_true", help="Return execution trace")
    return parser.parse_args()

def main() -> None:
    """
    CLI entry point for integer engine smoke testing.
    """
    args = parse_args()

    artifacts = load_quantized_artifacts(args.artifacts_dir)
    image_path = f"../data/busi/images/{args.image_sample}"
    x_fp = load_input_image_as_tensor(image_path).to(torch.device("cpu"))

    result = run_integer_inference(
        x_fp=x_fp,
        artifacts=artifacts,
        return_trace=args.return_trace,
    )
    print(result.keys())
    print("[int_engine] Integer inference completed.")
    print(f"[int_engine] Output int shape: {tuple(result['output_int'].shape)}")
    print(f"[int_engine] Output int dtype: {result['output_int'].dtype}")

    if "output_fp" in result:
        print(f"[int_engine] Output fp shape: {tuple(result['output_fp'].shape)}")

    if args.return_trace:
        print(f"[int_engine] Trace entries: {len(result['trace'])}")


if __name__ == "__main__":
    main()