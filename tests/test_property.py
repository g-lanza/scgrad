"""Hypothesis property tests: encoding round-trip, op closed forms, hardware, scc, tolerance.

Every property is checked over hypothesis-drawn inputs constrained to the
encodable range. Torch randomness is seeded from drawn integers and the
hypothesis settings use derandomize=True, so the whole file replays
identically on every run.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from scgrad.accuracy import tolerance
from scgrad.correlation import scc
from scgrad.encoding import SCConfig, decode, encode
from scgrad.hardware import make_source
from scgrad.ops import sc_add_tree, sc_mul

torch.manual_seed(0)

COMMON = settings(max_examples=50, deadline=None, derandomize=True)

ENCODINGS = ("unipolar", "bipolar")

bipolar_floats = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, width=32)
unit_floats = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, width=32)


def _config(encoding: str) -> SCConfig:
    return SCConfig(encoding=encoding, length=1024, source="sobol", seed=0)  # type: ignore[arg-type]


@COMMON
@given(encoding=st.sampled_from(ENCODINGS), raw=bipolar_floats)
def test_encode_decode_round_trip(encoding: str, raw: float) -> None:
    """Any value in the encodable range survives encode -> decode within 1e-6."""
    value = raw if encoding == "bipolar" else (raw + 1.0) / 2.0
    x = torch.tensor([value], dtype=torch.float64)
    decoded = decode(encode(x, _config(encoding)), descale=True)
    assert decoded.shape == x.shape
    assert (decoded - x).abs().max().item() <= 1e-6


@COMMON
@given(va=bipolar_floats, vb=bipolar_floats)
def test_sc_mul_is_exact_product_and_in_range(va: float, vb: float) -> None:
    """Bipolar sc_mul is the closed-form product, scale 1, physical value in range."""
    cfg = _config("bipolar")
    a = encode(torch.tensor([va], dtype=torch.float64), cfg)
    b = encode(torch.tensor([vb], dtype=torch.float64), cfg)
    out = sc_mul(a, b)
    assert out.scale == 1.0
    assert decode(out, descale=True).item() == pytest.approx(va * vb, abs=1e-12)
    assert abs(out.value.item()) <= 1.0


@COMMON
@given(values=st.lists(bipolar_floats, min_size=2, max_size=16))
def test_sc_add_tree_recovers_sum(values: list[float]) -> None:
    """A k-way MUX tree of scale-1 terms descales to the full sum; the wire stays in range."""
    cfg = _config("bipolar")
    terms = [encode(torch.tensor([v], dtype=torch.float64), cfg) for v in values]
    out = sc_add_tree(terms)
    k = len(values)
    assert out.scale == pytest.approx(1.0 / k)
    assert decode(out, descale=True).item() == pytest.approx(sum(values), abs=1e-5)
    assert abs(out.value.item()) <= 1.0 + 1e-12


@COMMON
@given(p=unit_floats, n=st.sampled_from([64, 256, 1024]), corr_id=st.integers(0, 7))
def test_sobol_empirical_probability(p: float, n: int, corr_id: int) -> None:
    """Sobol bitstream empirical P(bit=1) tracks p within 4/sqrt(n) for any p and small id."""
    cfg = SCConfig(encoding="unipolar", length=n, source="sobol", seed=0)
    src = make_source(cfg)
    bits = src.bits(torch.tensor([p], dtype=torch.float64), n, corr_id)
    assert bits.shape == (1, n)
    emp = bits.to(torch.float64).mean().item()
    assert abs(emp - p) <= 4.0 / math.sqrt(n)


@COMMON
@given(pa=unit_floats, pb=unit_floats, torch_seed=st.integers(0, 2**16 - 1))
def test_scc_bounded_for_random_bits(pa: float, pb: float, torch_seed: int) -> None:
    """SCC of arbitrary Bernoulli bit tensors always lands in [-1, 1] (tiny float slack)."""
    torch.manual_seed(torch_seed)
    a = torch.bernoulli(torch.full((4, 256), pa, dtype=torch.float64))
    b = torch.bernoulli(torch.full((4, 256), pb, dtype=torch.float64))
    s = scc(a, b)
    assert s.shape == (4,)
    assert s.min().item() >= -1.0 - 1e-9
    assert s.max().item() <= 1.0 + 1e-9


@COMMON
@given(
    n=st.integers(min_value=4, max_value=2048),
    step=st.integers(min_value=1, max_value=2048),
    depth=st.integers(min_value=1, max_value=4),
    encoding=st.sampled_from(ENCODINGS),
)
def test_tolerance_positive_and_monotone_in_n(n: int, step: int, depth: int, encoding: str) -> None:
    """tolerance(n, depth) is positive and strictly decreases as n grows."""
    t_small = tolerance(n, depth=depth, encoding=encoding)  # type: ignore[arg-type]
    t_large = tolerance(n + step, depth=depth, encoding=encoding)  # type: ignore[arg-type]
    assert t_small > 0.0
    assert t_large > 0.0
    assert t_small > t_large
