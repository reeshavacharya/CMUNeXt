# =========================================================
# GELU via NFGEN (Polynomial Approximation)
# =========================================================

import sys
import os
import math

import numpy as np
import sympy as sp
from dill.source import getsource

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from NFGen.main import generate_nonlinear_config
import NFGen.CodeTemplet.templet as temp
import NFGen.PerformanceModel.time_ops as to
import NFGen.code_generator as nf_cg


# Use the same default exponent as integer_inference.py
CONSTANT_EXPONENT = 16
# NFGEN GELU

# constant factors
PAI = 3.141592653589793
TAU_2 = 0.959502
ALPHA1 = 1.0
ALPHA2 = 1.6732632423543772848170429916717
LAMBDA = 1.0507009873554804934193349852946
E = 2.7182818
C1 = 0.044715
TAU_half = 1.7725
G = 0.5

platform = "Rep3"
profiler_file = 'NFGen/src/NFGen/PerformanceModel/' + platform + "_kmProfiler.pkl"


def _patched_code_generate(km_config,
                           profiler,
                           func,
                           basic_time,
                           code_templet,
                           basic_building_blocks,
                           save_file,
                           nick_name=None,
                           not_check=False,
                           code_language="python"):
    """Patched version of NFGen.code_generator.code_generate.

    Avoids the time_km formatting bug by skipping profiler-based
    time prediction and directly generating the Python approximation
    code using the provided template.
    """

    exec_code = getsource(func)
    if nick_name is None:
        func_name = exec_code[exec_code.index("def ") + 4:exec_code.index("(")]
    else:
        func_name = nick_name

    if code_language != "python":
        # Fallback to original implementation for non-Python targets.
        return nf_cg.code_generate(
            km_config,
            profiler,
            func,
            basic_time,
            code_templet,
            basic_building_blocks,
            save_file,
            nick_name=nick_name,
            not_check=not_check,
            code_language=code_language,
        )

    templet = code_templet.split("# insert here")
    templet[0] = templet[0].replace("general_non_linear_func", func_name)
    config_code = (
        "\n    breaks = " + str(km_config['breaks'][:-1]) +
        "\n    coeffA = " + str(km_config['coeffA']) +
        "\n    scaler = " + str(km_config['scaler'])
    )
    exec_code = templet[0] + config_code + templet[1] + "\n\n"

    with open(save_file, 'a') as f:
        f.write(exec_code)

    print("Write", func_name, "in", save_file, "SUCCESS!")
    return exec_code


# Monkeypatch NFGen's code generator at runtime to avoid touching
# the installed package files on disk.
nf_cg.code_generate = _patched_code_generate


def gelu(x):
    """Vectorized GELU implementation for NumPy arrays.

    NFGen will call this with NumPy arrays; use np.exp instead of
    math.exp to avoid scalar-only assumptions.
    """
    constant = math.sqrt(2.0 / PAI)
    x1 = constant * (x + C1 * x * x * x)
    ep = np.exp(x1)
    en = np.exp(-x1)
    return 0.5 * x * (1.0 + (ep - en) / (ep + en))


def setup_nfgen():
    n, f = 96, 48

    gelu_config = {
        "function": gelu,
        "range": (-5, 5),
        "k_max": 10,
        "tol": 1e-6,
        "ms": 1000,
        "zero_mask": 1e-6,
        "n": n,
        "f": f,
        "profiler": profiler_file,
        "code_templet": temp.templet_spdz,
        "code_language": "python",
        "config_file": "./gelu_approx.py",
        "time_dict": to.basic_time[platform],
        "derivative_flag": False,
    }

    generate_nonlinear_config(gelu_config)


if __name__ == "__main__":
    setup_nfgen()
