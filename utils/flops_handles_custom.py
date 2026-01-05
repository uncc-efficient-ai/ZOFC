# utils/flops_handles_custom.py
# Define per-op FLOP handle fns, and a helper to attach them to a FlopCountAnalysis.

from typing import Dict, Callable
from fvcore.nn.jit_handles import get_shape

# --------- tiny utils ----------
def _numel(x) -> int:
    s = get_shape(x)
    n = 1
    for d in s:
        n *= int(d)
    return n

# --------- handlers -------------
def softmax_flop_jit(inputs, outputs):
    # aten::softmax(Tensor self, int dim, bool half_to_float) -> Tensor
    x = inputs[0]
    return 3 * _numel(x)   # exp + reduce-sum + div ≈ 3 FLOPs/elt

def gelu_flop_jit(inputs, outputs):
    x = inputs[0]
    return 8 * _numel(x)   # tanh-approx; simple estimate

def linalg_vector_norm_flop_jit(inputs, outputs):
    x = inputs[0]
    return 3 * _numel(x)   # sum(x^2)+sqrt(+eps/div) upper bound

def add_flop_jit(inputs, outputs):
    return _numel(inputs[0])

def mul_flop_jit(inputs, outputs):
    return _numel(inputs[0])

def div_flop_jit(inputs, outputs):
    return _numel(inputs[0])

def clamp_min_flop_jit(inputs, outputs):
    return _numel(inputs[0])

def expand_as_flop_jit(inputs, outputs):
    return 0  # view/broadcast only

# Map of op schema names → handler. Include common overloads Torch uses.
_CUSTOM_HANDLES: Dict[str, Callable] = {
    "aten::softmax": softmax_flop_jit,
    "aten::softmax.int": softmax_flop_jit,

    "aten::gelu": gelu_flop_jit,

    "aten::linalg_vector_norm": linalg_vector_norm_flop_jit,

    "aten::add": add_flop_jit,
    "aten::add.Tensor": add_flop_jit,
    "aten::add.Scalar": add_flop_jit,

    "aten::mul": mul_flop_jit,
    "aten::mul.Tensor": mul_flop_jit,
    "aten::mul.Scalar": mul_flop_jit,

    "aten::div": div_flop_jit,
    "aten::div.Tensor": div_flop_jit,
    "aten::div.Scalar": div_flop_jit,

    "aten::clamp_min": clamp_min_flop_jit,
    "aten::expand_as": expand_as_flop_jit,
}

def apply_custom_jit_handles(fca) -> None:
    """
    Attach custom op handles directly to a FlopCountAnalysis instance.
    Works across fvcore versions — no global registration needed.
    """
    for op, fn in _CUSTOM_HANDLES.items():
        try:
            fca.set_op_handle(op, fn)
        except Exception:
            # Ignore if this op schema isn't present in this Torch build
            pass