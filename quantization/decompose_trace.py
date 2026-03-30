"""
decompose_trace.py

Purpose:
    Convert the raw module-based execution trace into a more explicit,
    verification-friendly execution plan.

Main transformations:
    1. Replace Residual wrapper ops with explicit Add ops.
    2. Insert explicit Concat ops before decoder fusion convs.
    3. Preserve all other traced ops as they are.
    4. Renumber execution_index so the final plan is clean and sequential.

Input:
    execution_trace.json

Output:
    decomposed_execution_trace.json

Why this file exists:
    The raw execution trace is module-based. Functional ops such as:
        - tensor addition inside Residual
        - torch.cat(...) in decoder skip fusion
    do not appear explicitly enough for later verifiable inference.

    This file enriches the trace into an execution plan without modifying
    the trained model architecture.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    """
    Load a JSON file.

    Args:
        path:
            Path to JSON file.

    Returns:
        Parsed Python object.
    """
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    """
    Save a Python object to JSON.

    Args:
        obj:
            JSON-serializable object.
        path:
            Output file path.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def build_concat_specs() -> Dict[str, Dict[str, Any]]:
    """
    Return the manual decoder concat mapping for CMUNeXt.

    The key is the fusion conv that consumes the concat output.
    The inserted Concat op will produce exactly the input tensor name used by
    that conv entry, typically something like:
        'Up_conv5.conv.0:input0'

    Returns:
        Dict mapping fusion-conv op name -> concat specification.

    Notes:
        These specs were derived from the execution trace and model structure.
    """
    return {
        "Up_conv5.conv.0": {
            "concat_name": "Up_conv5.concat",
            "inputs": [
                "encoder4.up.conv.0:out",
                "Up5.up.1:out",
            ],
        },
        "Up_conv4.conv.0": {
            "concat_name": "Up_conv4.concat",
            "inputs": [
                "encoder3.up.conv.0:out",
                "Up4.up.1:out",
            ],
        },
        "Up_conv3.conv.0": {
            "concat_name": "Up_conv3.concat",
            "inputs": [
                "encoder2.up.conv.0:out",
                "Up3.up.1:out",
            ],
        },
        "Up_conv2.conv.0": {
            "concat_name": "Up_conv2.concat",
            "inputs": [
                "encoder1.up.conv.0:out",
                "Up2.up.1:out",
            ],
        },
    }


def make_add_entry(
    residual_entry: Dict[str, Any],
    previous_entry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Replace one Residual wrapper entry with an explicit Add op entry.

    Args:
        residual_entry:
            Raw trace entry whose op_type is 'Residual'
        previous_entry:
            The immediately preceding entry in execution order.
            For current CMUNeXt trace structure, this is assumed to be the
            residual branch output.

    Returns:
        New explicit Add entry.

    Important assumption:
        For the current raw trace, Residual is emitted after the final op of
        the residual branch. Therefore:
            skip input = residual_entry["input_tensor_names"][0]
            branch output = previous_entry["output_tensor_name"]
    """
    if len(residual_entry.get("input_tensor_names", [])) == 0:
        raise ValueError(
            f"Residual entry '{residual_entry.get('op_name')}' has no input_tensor_names"
        )

    add_entry = {
        "execution_index": residual_entry["execution_index"],
        "input_shapes": [],  # optional; can be filled later if desired
        "input_tensor_names": [
            residual_entry["input_tensor_names"][0],
            previous_entry["output_tensor_name"],
        ],
        "op_name": f"{residual_entry['op_name']}.add",
        "op_type": "Add",
        "output_shape": residual_entry.get("output_shape", []),
        "output_tensor_name": residual_entry["output_tensor_name"],
    }

    return add_entry


def make_concat_entry(
    target_conv_entry: Dict[str, Any],
    concat_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create an explicit Concat entry to be inserted before a decoder fusion conv.

    Args:
        target_conv_entry:
            The fusion conv trace entry, e.g. 'Up_conv5.conv.0'
        concat_spec:
            Manual spec containing concat_name and the two input tensor names.

    Returns:
        New explicit Concat entry.

    Notes:
        The output tensor name of the Concat entry is set to the conv's existing
        input placeholder, e.g. 'Up_conv5.conv.0:input0', so the downstream conv
        can remain unchanged.
    """
    if len(target_conv_entry.get("input_tensor_names", [])) == 0:
        raise ValueError(
            f"Target conv entry '{target_conv_entry.get('op_name')}' has no input_tensor_names"
        )

    concat_entry = {
        "execution_index": target_conv_entry["execution_index"],
        "input_shapes": [],  # optional; can be filled later if desired
        "input_tensor_names": concat_spec["inputs"],
        "op_name": concat_spec["concat_name"],
        "op_type": "Concat",
        "output_shape": target_conv_entry.get("input_shapes", [[]])[0]
        if len(target_conv_entry.get("input_shapes", [])) > 0
        else [],
        "output_tensor_name": target_conv_entry["input_tensor_names"][0],
    }

    return concat_entry


def renumber_execution_indices(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Renumber execution_index fields so they are 0,1,2,... in final order.

    Args:
        entries:
            Decomposed execution plan entries in desired order.

    Returns:
        Same entries with clean sequential execution_index values.
    """
    for idx, entry in enumerate(entries):
        entry["execution_index"] = idx
    return entries


def decompose_trace(
    raw_trace: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert the raw execution trace into a decomposed execution plan.

    Transformations:
        - Residual -> Add
        - insert Concat before known decoder fusion convs
        - preserve all other entries

    Args:
        raw_trace:
            Raw execution trace loaded from execution_trace.json

    Returns:
        Decomposed execution plan as a list of dict entries.

    Important:
        This function assumes raw_trace is already in execution order.
    """
    concat_specs = build_concat_specs()
    decomposed: List[Dict[str, Any]] = []

    for i, entry in enumerate(raw_trace):
        op_name = entry.get("op_name", "")
        op_type = entry.get("op_type", "")

        # ------------------------------------------------------------
        # 1. Insert explicit Concat before known decoder fusion convs
        # ------------------------------------------------------------
        if op_name in concat_specs:
            concat_entry = make_concat_entry(
                target_conv_entry=entry,
                concat_spec=concat_specs[op_name],
            )
            decomposed.append(concat_entry)

        # ------------------------------------------------------------
        # 2. Replace Residual wrapper with explicit Add
        # ------------------------------------------------------------
        if op_type == "Residual":
            if i == 0:
                raise RuntimeError(
                    f"Cannot decompose Residual entry '{op_name}' at position 0 "
                    f"because there is no previous entry"
                )

            previous_entry = raw_trace[i - 1]
            add_entry = make_add_entry(
                residual_entry=entry,
                previous_entry=previous_entry,
            )
            decomposed.append(add_entry)
            continue

        # ------------------------------------------------------------
        # 3. Keep everything else unchanged
        # ------------------------------------------------------------
        decomposed.append(dict(entry))

    # Clean sequential execution order
    decomposed = renumber_execution_indices(decomposed)
    return decomposed


def print_plan_summary(
    raw_trace: List[Dict[str, Any]],
    decomposed_trace: List[Dict[str, Any]],
) -> None:
    """
    Print a short summary of the decomposition.

    Args:
        raw_trace:
            Original raw trace
        decomposed_trace:
            Final decomposed plan
    """
    raw_types: Dict[str, int] = {}
    new_types: Dict[str, int] = {}

    for entry in raw_trace:
        raw_types[entry["op_type"]] = raw_types.get(entry["op_type"], 0) + 1

    for entry in decomposed_trace:
        new_types[entry["op_type"]] = new_types.get(entry["op_type"], 0) + 1

    print(f"[decompose_trace] Raw trace entries: {len(raw_trace)}")
    print(f"[decompose_trace] Decomposed entries: {len(decomposed_trace)}")

    print("[decompose_trace] Raw op counts:")
    for k in sorted(raw_types):
        print(f"  {k}: {raw_types[k]}")

    print("[decompose_trace] Decomposed op counts:")
    for k in sorted(new_types):
        print(f"  {k}: {new_types[k]}")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args for trace decomposition.
    """
    parser = argparse.ArgumentParser(
        description="Decompose raw execution trace into explicit Add/Concat plan"
    )

    parser.add_argument(
        "--input-trace",
        type=str,
        required=True,
        help="Path to raw execution_trace.json",
    )
    parser.add_argument(
        "--output-trace",
        type=str,
        required=True,
        help="Path to decomposed_execution_trace.json",
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()

    print(f"[decompose_trace] Loading raw trace: {args.input_trace}")
    raw_trace = load_json(args.input_trace)

    if not isinstance(raw_trace, list):
        raise TypeError("execution_trace.json must contain a list of trace entries")

    decomposed_trace = decompose_trace(raw_trace)

    print(f"[decompose_trace] Saving decomposed trace: {args.output_trace}")
    save_json(decomposed_trace, args.output_trace)

    print_plan_summary(raw_trace, decomposed_trace)
    print("[decompose_trace] Done.")


if __name__ == "__main__":
    main()