"""Tests for scgrad.correlation: SCC measurement, tracking, and the differentiable proxy.

The central claim under test: the differentiable collision penalty
(correlated_multiply_error) moves with the real hardware error measured
on actual shared-generator bitstreams, so minimizing it during training
reduces the error the exact path would show.
"""

import torch

from scgrad.correlation import (
    CorrelationTracker,
    MultiplyEvent,
    correlated_multiply_error,
    correlation_loss,
    scc,
)
from scgrad.encoding import SCConfig, encode
from scgrad.hardware import make_source
from scgrad.layers import SCLinear
from scgrad.ops import sc_mul

N = 4096


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Rank positions of a 1-d tensor (0 .. n-1), arbitrary order on ties."""
    order = torch.argsort(x)
    ranks = torch.empty(x.shape[0], dtype=torch.float64)
    ranks[order] = torch.arange(x.shape[0], dtype=torch.float64)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation of two 1-d tensors, plain torch."""
    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def test_scc_identical_streams_is_plus_one() -> None:
    config = SCConfig(encoding="bipolar", length=N, source="sobol", seed=0)
    source = make_source(config)
    p = torch.tensor(0.5)
    a = source.bits(p, N, corr_id=0)
    b = source.bits(p, N, corr_id=0)
    assert scc(a, b).item() > 0.999


def test_scc_independent_streams_near_zero() -> None:
    config = SCConfig(encoding="bipolar", length=N, source="sobol", seed=0)
    source = make_source(config)
    p = torch.tensor([0.3, 0.5, 0.7])
    a = source.bits(p, N, corr_id=0)
    b = source.bits(p, N, corr_id=1)
    assert scc(a, b).abs().max().item() < 0.1


def test_scc_degenerate_streams_zero() -> None:
    config = SCConfig(encoding="bipolar", length=N, source="sobol", seed=0)
    source = make_source(config)
    ones = source.bits(torch.tensor(1.0), N, corr_id=0)
    zeros = source.bits(torch.tensor(0.0), N, corr_id=1)
    half = source.bits(torch.tensor(0.5), N, corr_id=2)
    assert bool(ones.all())
    assert not bool(zeros.any())
    assert scc(ones, half).item() == 0.0
    assert scc(ones, ones).item() == 0.0
    assert scc(zeros, half).item() == 0.0


def test_scc_complementary_streams_is_minus_one() -> None:
    config = SCConfig(encoding="bipolar", length=N, source="sobol", seed=0)
    source = make_source(config)
    a = source.bits(torch.tensor(0.3), N, corr_id=0)
    b = ~a
    assert scc(a, b).item() < -0.999


def test_tracker_records_bare_sc_mul() -> None:
    config = SCConfig(encoding="bipolar", length=256, seed=0)
    a = encode(torch.tensor([0.4]), config)
    b = encode(torch.tensor([-0.6]), config)
    with CorrelationTracker() as tracker:
        sc_mul(a, b)
    assert len(tracker.events) == 1
    event = tracker.events[0]
    assert event.corr_id_a == a.corr_id
    assert event.corr_id_b == b.corr_id
    assert torch.allclose(event.p_a, a.probabilities())
    assert torch.allclose(event.p_b, b.probabilities())


def test_tracker_records_inside_sclinear_forward() -> None:
    torch.manual_seed(0)
    config = SCConfig(encoding="bipolar", length=256, seed=0)
    layer = SCLinear(3, 2, config=config)
    x = torch.rand(1, 3) * 2.0 - 1.0
    with CorrelationTracker() as tracker:
        layer(x)
    assert len(tracker.events) == 1
    event = tracker.events[0]
    assert event.corr_id_a == layer.input_corr_id
    assert event.corr_id_b == layer.weight_corr_id


def test_collisions_under_shared_rng_budget() -> None:
    config = SCConfig(encoding="bipolar", length=256, seed=0, n_rngs=1)
    a = encode(torch.tensor([0.4]), config)
    b = encode(torch.tensor([-0.6]), config)
    assert a.corr_id != b.corr_id
    with CorrelationTracker() as tracker:
        sc_mul(a, b)
    assert len(tracker.collisions()) == 1


def test_no_collisions_with_unlimited_rngs() -> None:
    config = SCConfig(encoding="bipolar", length=256, seed=0, n_rngs=None)
    a = encode(torch.tensor([0.4]), config)
    b = encode(torch.tensor([-0.6]), config)
    with CorrelationTracker() as tracker:
        sc_mul(a, b)
    assert tracker.collisions() == []
    with CorrelationTracker() as self_tracker:
        sc_mul(a, a)
    assert len(self_tracker.collisions()) == 1


def test_correlated_multiply_error_unipolar_closed_form() -> None:
    config = SCConfig(encoding="unipolar", length=256, seed=0)
    event = MultiplyEvent(
        corr_id_a=0,
        corr_id_b=1,
        p_a=torch.tensor([0.3, 0.5]),
        p_b=torch.tensor([0.8, 0.5]),
        encoding="unipolar",
        config=config,
    )
    err = correlated_multiply_error(event)
    expected = torch.tensor([0.3 - 0.24, 0.5 - 0.25])
    assert torch.allclose(err, expected, atol=1e-7)


def test_correlated_multiply_error_bipolar_closed_form() -> None:
    config = SCConfig(encoding="bipolar", length=256, seed=0)
    event = MultiplyEvent(
        corr_id_a=0,
        corr_id_b=1,
        p_a=torch.tensor([0.75, 0.8]),
        p_b=torch.tensor([0.25, 0.8]),
        encoding="bipolar",
        config=config,
    )
    err = correlated_multiply_error(event)
    expected = torch.tensor([0.25, 1.0 - 0.36])
    assert torch.allclose(err, expected, atol=1e-6)


def test_proxy_rank_correlates_with_exact_error_bipolar() -> None:
    """The proxy must move with reality: rank-correlate against measured XNOR error."""
    torch.manual_seed(7)
    config = SCConfig(encoding="bipolar", length=N, source="sobol", seed=0, n_rngs=1)
    source = make_source(config)
    va = torch.rand(24, dtype=torch.float64) * 1.8 - 0.9
    vb = torch.rand(24, dtype=torch.float64) * 1.8 - 0.9
    pa = (va + 1.0) / 2.0
    pb = (vb + 1.0) / 2.0
    bits_a = source.bits(pa, N, corr_id=0)
    bits_b = source.bits(pb, N, corr_id=1)
    assert config.rng_index(0) == config.rng_index(1)
    xnor = bits_a == bits_b
    v_out = 2.0 * xnor.to(torch.float64).mean(dim=-1) - 1.0
    exact_err = (v_out - va * vb).abs()
    event = MultiplyEvent(
        corr_id_a=0, corr_id_b=1, p_a=pa, p_b=pb, encoding="bipolar", config=config
    )
    proxy_err = correlated_multiply_error(event)
    assert _spearman(proxy_err, exact_err) > 0.8


def test_proxy_rank_correlates_with_exact_error_unipolar() -> None:
    torch.manual_seed(11)
    config = SCConfig(encoding="unipolar", length=N, source="sobol", seed=0, n_rngs=1)
    source = make_source(config)
    pa = torch.rand(24, dtype=torch.float64) * 0.9 + 0.05
    pb = torch.rand(24, dtype=torch.float64) * 0.9 + 0.05
    bits_a = source.bits(pa, N, corr_id=0)
    bits_b = source.bits(pb, N, corr_id=1)
    p_out = (bits_a & bits_b).to(torch.float64).mean(dim=-1)
    exact_err = (p_out - pa * pb).abs()
    event = MultiplyEvent(
        corr_id_a=0, corr_id_b=1, p_a=pa, p_b=pb, encoding="unipolar", config=config
    )
    proxy_err = correlated_multiply_error(event)
    assert _spearman(proxy_err, exact_err) > 0.8


def test_correlation_loss_zero_without_collisions() -> None:
    config = SCConfig(encoding="bipolar", length=256, seed=0, n_rngs=None)
    a = encode(torch.tensor([0.4]), config)
    b = encode(torch.tensor([-0.6]), config)
    with CorrelationTracker() as tracker:
        sc_mul(a, b)
    loss = correlation_loss(tracker)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert loss.item() == 0.0
    empty = correlation_loss(CorrelationTracker())
    assert empty.item() == 0.0


def test_correlation_loss_positive_and_exact_on_collision() -> None:
    config = SCConfig(encoding="bipolar", length=256, seed=0, n_rngs=1)
    a = encode(torch.tensor([0.5]), config)
    b = encode(torch.tensor([-0.5]), config)
    with CorrelationTracker() as tracker:
        sc_mul(a, b)
    loss = correlation_loss(tracker)
    assert loss.item() > 0.0
    assert abs(loss.item() - 0.25) < 1e-6


def test_correlation_loss_gradients_flow_to_weights() -> None:
    torch.manual_seed(3)
    config = SCConfig(encoding="bipolar", length=256, seed=0, n_rngs=1)
    layer = SCLinear(4, 3, config=config)
    x = torch.rand(2, 4) * 2.0 - 1.0
    with CorrelationTracker() as tracker:
        layer(x)
    assert len(tracker.collisions()) == 1
    loss = correlation_loss(tracker)
    assert loss.requires_grad
    assert loss.item() > 0.0
    loss.backward()
    grad = layer.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0.0
