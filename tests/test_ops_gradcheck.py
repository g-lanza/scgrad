"""Gradcheck gate and behavior tests for scgrad.ops primitives.

The hard gate: float64 torch.autograd.gradcheck (eps=1e-6, atol=1e-4)
for MulFunction, AddFunction, and AddTreeFunction, plus gradgradcheck
for MulFunction. Behavior tests pin the value-space semantics: products,
MUX scaled averages, scale propagation, validation errors, and the
correlation tracker recording rng-colliding multiplies.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from scgrad.correlation import CorrelationTracker, correlation_loss
from scgrad.encoding import SCConfig, SCEncodingError, decode, encode
from scgrad.ops import (
    AddFunction,
    AddTreeFunction,
    MulFunction,
    sc_add,
    sc_add_tree,
    sc_mul,
)

EPS = 1e-6
ATOL = 1e-4


def bipolar_config(**overrides: object) -> SCConfig:
    """A small deterministic bipolar config for behavior tests."""
    defaults: dict[str, object] = {"encoding": "bipolar", "length": 256, "seed": 7}
    defaults.update(overrides)
    return SCConfig(**defaults)  # type: ignore[arg-type]


def rand_in_range(shape: tuple[int, ...], lo: float, hi: float) -> torch.Tensor:
    """Uniform tensor strictly inside (lo, hi) so encode's clamp is the identity."""
    return lo + (hi - lo) * (0.05 + 0.9 * torch.rand(shape))


def test_gradcheck_mul_function() -> None:
    """MulFunction.apply passes float64 gradcheck on two small tensors."""
    torch.manual_seed(0)
    a = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    b = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    assert gradcheck(MulFunction.apply, (a, b), eps=EPS, atol=ATOL)


def test_gradgradcheck_mul_function() -> None:
    """MulFunction.apply passes float64 gradgradcheck (second-order)."""
    torch.manual_seed(1)
    a = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    b = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    assert gradgradcheck(MulFunction.apply, (a, b), eps=EPS, atol=ATOL)


def test_gradcheck_add_function() -> None:
    """AddFunction.apply passes float64 gradcheck with a non-tensor select_p."""
    torch.manual_seed(2)
    a = torch.rand(3, dtype=torch.float64, requires_grad=True)
    b = torch.rand(3, dtype=torch.float64, requires_grad=True)
    assert gradcheck(AddFunction.apply, (a, b, 0.3), eps=EPS, atol=ATOL)


def test_gradcheck_add_tree_function() -> None:
    """AddTreeFunction.apply passes float64 gradcheck on a stacked term tensor."""
    torch.manual_seed(3)
    stacked = torch.rand(5, 2, 3, dtype=torch.float64, requires_grad=True)
    assert gradcheck(AddTreeFunction.apply, (stacked,), eps=EPS, atol=ATOL)


def test_bipolar_sc_mul_value_is_elementwise_product() -> None:
    """Bipolar sc_mul produces the elementwise product in value space."""
    torch.manual_seed(4)
    cfg = bipolar_config()
    xa = rand_in_range((4,), -1.0, 1.0)
    xb = rand_in_range((4,), -1.0, 1.0)
    out = sc_mul(encode(xa, cfg), encode(xb, cfg))
    assert torch.allclose(out.value, xa * xb)
    assert out.scale == pytest.approx(1.0)


def test_unipolar_sc_mul_value_is_elementwise_product() -> None:
    """Unipolar sc_mul produces the elementwise product in value space."""
    torch.manual_seed(5)
    cfg = bipolar_config(encoding="unipolar")
    xa = rand_in_range((4,), 0.0, 1.0)
    xb = rand_in_range((4,), 0.0, 1.0)
    out = sc_mul(encode(xa, cfg), encode(xb, cfg))
    assert torch.allclose(out.value, xa * xb)
    assert out.scale == pytest.approx(1.0)


def test_sc_add_value_is_scaled_average_and_decode_recovers_sum() -> None:
    """Default sc_add halves the sum physically; decode recovers va + vb at unit scales."""
    torch.manual_seed(6)
    cfg = bipolar_config()
    xa = rand_in_range((4,), -1.0, 1.0)
    xb = rand_in_range((4,), -1.0, 1.0)
    out = sc_add(encode(xa, cfg), encode(xb, cfg))
    assert torch.allclose(out.value, 0.5 * (xa + xb))
    assert out.scale == pytest.approx(0.5)
    assert torch.allclose(decode(out, descale=True), xa + xb)


def test_sc_add_custom_select_p_value() -> None:
    """sc_add with select_p=0.3 produces the 0.3/0.7 weighted average."""
    torch.manual_seed(7)
    cfg = bipolar_config()
    xa = rand_in_range((3,), -1.0, 1.0)
    xb = rand_in_range((3,), -1.0, 1.0)
    out = sc_add(encode(xa, cfg), encode(xb, cfg), select_p=0.3)
    assert torch.allclose(out.value, 0.3 * xa + 0.7 * xb)
    assert out.scale == pytest.approx(0.5)


def test_sc_add_tree_mean_scale_and_decode() -> None:
    """sc_add_tree of k unit-scale terms: value=mean, scale=1/k, decode=full sum."""
    torch.manual_seed(8)
    cfg = bipolar_config()
    k = 5
    values = [rand_in_range((3,), -1.0, 1.0) for _ in range(k)]
    out = sc_add_tree([encode(v, cfg) for v in values])
    stacked = torch.stack(values, dim=0)
    assert torch.allclose(out.value, stacked.mean(dim=0))
    assert out.scale == pytest.approx(1.0 / k)
    assert torch.allclose(decode(out, descale=True), stacked.sum(dim=0))


def test_scale_propagates_through_mul_then_tree() -> None:
    """A multiply-accumulate (mul then tree) decodes to the exact dot-product sum."""
    torch.manual_seed(9)
    cfg = bipolar_config()
    k = 4
    xs = [rand_in_range((2,), -1.0, 1.0) for _ in range(k)]
    ws = [rand_in_range((2,), -1.0, 1.0) for _ in range(k)]
    pairs = list(zip(xs, ws, strict=True))
    products = [sc_mul(encode(x, cfg), encode(w, cfg)) for x, w in pairs]
    out = sc_add_tree(products)
    assert out.scale == pytest.approx(1.0 / k)
    expected = torch.stack([x * w for x, w in pairs], dim=0).sum(dim=0)
    assert torch.allclose(decode(out, descale=True), expected)


def test_scale_propagates_through_add_then_mul() -> None:
    """Multiplying a scaled MUX sum by a fresh operand keeps the scale; decode is exact."""
    torch.manual_seed(10)
    cfg = bipolar_config()
    xa = rand_in_range((3,), -1.0, 1.0)
    xb = rand_in_range((3,), -1.0, 1.0)
    xc = rand_in_range((3,), -1.0, 1.0)
    summed = sc_add(encode(xa, cfg), encode(xb, cfg))
    out = sc_mul(summed, encode(xc, cfg))
    assert out.scale == pytest.approx(0.5)
    assert torch.allclose(decode(out, descale=True), (xa + xb) * xc)


def test_shape_mismatch_raises() -> None:
    """sc_mul and sc_add reject same-encoding operands with different shapes."""
    torch.manual_seed(11)
    cfg = bipolar_config()
    a = encode(rand_in_range((2,), -1.0, 1.0), cfg)
    b = encode(rand_in_range((3,), -1.0, 1.0), cfg)
    with pytest.raises(SCEncodingError):
        sc_mul(a, b)
    with pytest.raises(SCEncodingError):
        sc_add(a, b)


def test_length_mismatch_raises() -> None:
    """Operands from configs with different bitstream lengths cannot be combined."""
    torch.manual_seed(12)
    x = rand_in_range((2,), -1.0, 1.0)
    a = encode(x, bipolar_config(length=128))
    b = encode(x, bipolar_config(length=256))
    with pytest.raises(SCEncodingError):
        sc_mul(a, b)
    with pytest.raises(SCEncodingError):
        sc_add(a, b)


def test_encoding_mismatch_raises() -> None:
    """Mixing bipolar and unipolar operands raises SCEncodingError."""
    torch.manual_seed(13)
    x = rand_in_range((2,), 0.0, 1.0)
    a = encode(x, bipolar_config())
    b = encode(x, bipolar_config(encoding="unipolar"))
    with pytest.raises(SCEncodingError):
        sc_mul(a, b)


@pytest.mark.parametrize("select_p", [0.0, 1.0, -0.2, 1.5])
def test_select_p_outside_open_interval_raises(select_p: float) -> None:
    """sc_add rejects select_p values outside the open interval (0, 1)."""
    torch.manual_seed(14)
    cfg = bipolar_config()
    x = rand_in_range((2,), -1.0, 1.0)
    a = encode(x, cfg)
    b = encode(-x, cfg)
    with pytest.raises(SCEncodingError):
        sc_add(a, b, select_p=select_p)


def test_empty_add_tree_raises() -> None:
    """sc_add_tree with no terms raises SCEncodingError."""
    with pytest.raises(SCEncodingError):
        sc_add_tree([])


def test_correlated_multiply_is_recorded_and_penalized() -> None:
    """With n_rngs=1, a tracked sc_mul logs a collision and the loss is positive."""
    cfg = bipolar_config(n_rngs=1)
    a = encode(torch.tensor([0.5, -0.25]), cfg)
    b = encode(torch.tensor([-0.25, 0.75]), cfg)
    with CorrelationTracker() as tracker:
        sc_mul(a, b)
    assert len(tracker.collisions()) == 1
    loss = correlation_loss(tracker)
    assert loss.item() > 0.0


def test_independent_rngs_record_no_collision() -> None:
    """With unlimited generators, tracked multiplies record events but no collisions."""
    cfg = bipolar_config(n_rngs=None)
    a = encode(torch.tensor([0.5, -0.25]), cfg)
    b = encode(torch.tensor([-0.25, 0.75]), cfg)
    with CorrelationTracker() as tracker:
        sc_mul(a, b)
    assert len(tracker.events) == 1
    assert tracker.collisions() == []
    assert correlation_loss(tracker).item() == pytest.approx(0.0)
