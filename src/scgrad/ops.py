"""The differentiable SC primitives: sc_mul, sc_add, sc_add_tree.

Forward passes are the closed-form expected values of the corresponding
gate circuits on independent streams, so they are exact in expectation
and smooth; backwards are hand-written and gated by float64 gradcheck.

The two correctness traps, handled here once:

Bipolar multiply is XNOR. From p* = p_a p_b + (1 - p_a)(1 - p_b)
(Gaines 1968) the value map v = 2p - 1 collapses it to v = v_a * v_b, so
both encodings multiply in value space, with independence as the
precondition (sc_mul records every multiply with the correlation
tracker; collisions are penalized by correlation.correlation_loss and
shown truthfully by the exact path).

MUX addition is a scaled average, not a sum. A k-input uniform MUX
outputs (1/k) * sum(v_i): that scaled value is the physical wire
quantity and is kept in SCNumber.value; the accumulated factor is kept
in SCNumber.scale. The rule for composing scales: for a MUX with select
weights w_i over inputs with scales s_i, the output scale is
mean_i(w_i * s_i). When every per-term factor w_i * s_i agrees (the
balanced case), decode(descale=True) recovers the exact intended sum;
with mismatched factors it is a weighted approximation, noted in the
docstrings below.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from scgrad.correlation import record_multiply
from scgrad.encoding import SCEncodingError, SCNumber, require_same_encoding


class MulFunction(torch.autograd.Function):
    """Elementwise product in value space (AND/XNOR expectation); product-rule backward."""

    @staticmethod
    def forward(ctx: Any, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        return grad_out * b, grad_out * a


class AddFunction(torch.autograd.Function):
    """Two-input MUX expectation: select_p * a + (1 - select_p) * b; linear backward."""

    @staticmethod
    def forward(ctx: Any, a: Tensor, b: Tensor, select_p: float) -> Tensor:
        ctx.select_p = select_p
        return select_p * a + (1.0 - select_p) * b

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, Tensor, None]:
        sp = ctx.select_p
        return grad_out * sp, grad_out * (1.0 - sp), None


class AddTreeFunction(torch.autograd.Function):
    """k-input uniform MUX expectation: mean over the stacked term dimension."""

    @staticmethod
    def forward(ctx: Any, stacked: Tensor) -> Tensor:
        ctx.k = stacked.shape[0]
        return stacked.mean(dim=0)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> Tensor:
        k: int = ctx.k
        return grad_out.unsqueeze(0).expand(k, *grad_out.shape) / k


def _check_compatible(a: SCNumber, b: SCNumber) -> None:
    require_same_encoding(a, b)
    if a.config.length != b.config.length:
        raise SCEncodingError(
            f"cannot mix bitstream lengths: {a.config.length} vs {b.config.length}"
        )
    if a.value.shape != b.value.shape:
        raise SCEncodingError(
            f"sc ops require same-shape operands: {tuple(a.value.shape)} vs {tuple(b.value.shape)}"
        )


def sc_mul(a: SCNumber, b: SCNumber) -> SCNumber:
    """Multiply two SCNumbers: one AND (unipolar) or XNOR (bipolar) gate per bit.

    Exact in expectation for independent streams. The multiply is
    recorded with the active correlation tracker; if the two streams
    share a generator under the config's randomness budget, the approx
    path still returns the intended product while the recorded event
    carries the collision cost (the exact path shows the real degraded
    result). Scales multiply (a multiply of physical values multiplies
    intents and scales alike; with a fresh operand of scale 1 the scale
    is unchanged). The output is a new stream with a fresh corr_id.
    """
    _check_compatible(a, b)
    record_multiply(a, b)
    value: Tensor = MulFunction.apply(a.value, b.value)  # type: ignore[no-untyped-call]
    return SCNumber(value, a.config, scale=a.scale * b.scale)


def sc_add(a: SCNumber, b: SCNumber, *, select_p: float = 0.5) -> SCNumber:
    """Add two SCNumbers with a MUX: the physical result is a scaled average.

    The output value is select_p * v_a + (1 - select_p) * v_b. The output
    scale follows the mean-of-term-factors rule (module docstring): with
    the default select_p = 0.5 and equal input scales s, the output scale
    is s / 2 and decode(descale=True) recovers v_a + v_b exactly. With
    unequal select weights or mismatched input scales the recovery is the
    correspondingly weighted combination.
    """
    _check_compatible(a, b)
    if not 0.0 < select_p < 1.0:
        raise SCEncodingError(f"select_p must be in (0, 1), got {select_p}")
    value: Tensor = AddFunction.apply(a.value, b.value, select_p)  # type: ignore[no-untyped-call]
    scale = (select_p * a.scale + (1.0 - select_p) * b.scale) / 2.0
    return SCNumber(value, a.config, scale=scale)


def sc_add_tree(terms: list[SCNumber]) -> SCNumber:
    """k-way MUX accumulation: physical output is the mean of the terms.

    Implemented as a uniform k-way selector, which is exact for any k
    (a balanced binary MUX tree realizes the same distribution for
    powers of two; for other k the uniform selector is the standard
    generalization and avoids padding bias). Output scale is
    mean_i(s_i) / k, so with a common input scale s the output scale is
    s / k and decode(descale=True) recovers the full sum. This is the
    accumulator layers use.
    """
    if not terms:
        raise SCEncodingError("sc_add_tree requires at least one term")
    first = terms[0]
    for t in terms[1:]:
        _check_compatible(first, t)
    stacked = torch.stack([t.value for t in terms], dim=0)
    value: Tensor = AddTreeFunction.apply(stacked)  # type: ignore[no-untyped-call]
    k = len(terms)
    scale = sum(t.scale for t in terms) / k / k
    return SCNumber(value, first.config, scale=scale)
