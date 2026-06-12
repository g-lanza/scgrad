"""scgrad: differentiable stochastic-computing primitives for PyTorch.

The public API surface, kept deliberately small. Everything here is
documented in its home module; docs/theory.md derives the math.
"""

from scgrad.accuracy import accuracy_estimator
from scgrad.correlation import CorrelationTracker, correlation_loss
from scgrad.encoding import SCConfig, SCNumber, decode, encode
from scgrad.eval_exact import evaluate_exact, evaluate_float
from scgrad.layers import SCConv2d, SCLinear, SCReLU, sc_relu
from scgrad.ops import sc_add, sc_add_tree, sc_mul

__all__ = [
    "CorrelationTracker",
    "SCConfig",
    "SCConv2d",
    "SCLinear",
    "SCNumber",
    "SCReLU",
    "accuracy_estimator",
    "correlation_loss",
    "decode",
    "encode",
    "evaluate_exact",
    "evaluate_float",
    "sc_add",
    "sc_add_tree",
    "sc_mul",
    "sc_relu",
]
