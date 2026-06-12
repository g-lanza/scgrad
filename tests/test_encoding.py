"""Tests for scgrad.encoding: round trips, clamping, STE, conversions, corr_ids."""

from __future__ import annotations

import pytest
import torch

from scgrad.encoding import (
    SCConfig,
    SCEncodingError,
    SCNumber,
    decode,
    encode,
    fresh_corr_id,
    probability_to_value,
    to_bipolar,
    to_unipolar,
    value_to_probability,
)
from scgrad.ops import sc_add, sc_mul

BIPOLAR = SCConfig(encoding="bipolar", length=256, seed=0)
UNIPOLAR = SCConfig(encoding="unipolar", length=256, seed=0)


def test_round_trip_bipolar() -> None:
    """encode then decode returns in-range bipolar values exactly."""
    torch.manual_seed(0)
    x = torch.rand(4, 3) * 2.0 - 1.0
    s = encode(x, BIPOLAR)
    assert s.scale == 1.0
    torch.testing.assert_close(decode(s), x, rtol=0.0, atol=0.0)


def test_round_trip_unipolar() -> None:
    """encode then decode returns in-range unipolar values exactly."""
    torch.manual_seed(1)
    x = torch.rand(5)
    s = encode(x, UNIPOLAR)
    assert s.scale == 1.0
    torch.testing.assert_close(decode(s), x, rtol=0.0, atol=0.0)


def test_clamp_at_bounds_is_identity() -> None:
    """Values exactly at the range bounds pass through unchanged."""
    xb = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(encode(xb, BIPOLAR).value, xb, rtol=0.0, atol=0.0)
    xu = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(encode(xu, UNIPOLAR).value, xu, rtol=0.0, atol=0.0)


def test_clamp_beyond_bounds_bipolar() -> None:
    """Out-of-range bipolar inputs clamp to [-1, 1]."""
    x = torch.tensor([-3.0, -1.0, 0.25, 1.0, 7.5])
    expected = torch.tensor([-1.0, -1.0, 0.25, 1.0, 1.0])
    torch.testing.assert_close(encode(x, BIPOLAR).value, expected, rtol=0.0, atol=0.0)


def test_clamp_beyond_bounds_unipolar() -> None:
    """Out-of-range unipolar inputs clamp to [0, 1]."""
    x = torch.tensor([-0.5, 0.0, 0.5, 1.0, 2.0])
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    torch.testing.assert_close(encode(x, UNIPOLAR).value, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("config", [BIPOLAR, UNIPOLAR])
def test_ste_gradient_passes_through_clamp(config: SCConfig) -> None:
    """Gradient of encode(x).value.sum() is 1 even where x is clamped."""
    x = torch.tensor([-5.0, -0.3, 0.4, 5.0], requires_grad=True)
    encode(x, config).value.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0.0, atol=0.0)


def test_encode_rejects_non_tensor() -> None:
    """encode raises SCEncodingError for non-Tensor input."""
    with pytest.raises(SCEncodingError, match="expects a Tensor"):
        encode(0.5, BIPOLAR)  # type: ignore[arg-type]


def test_mixed_encodings_raise_in_mul() -> None:
    """sc_mul on mixed encodings raises SCEncodingError."""
    a = encode(torch.tensor([0.5]), BIPOLAR)
    b = encode(torch.tensor([0.5]), UNIPOLAR)
    with pytest.raises(SCEncodingError, match="mix encodings"):
        sc_mul(a, b)


def test_mixed_encodings_raise_in_add() -> None:
    """sc_add on mixed encodings raises SCEncodingError."""
    a = encode(torch.tensor([0.5]), BIPOLAR)
    b = encode(torch.tensor([0.5]), UNIPOLAR)
    with pytest.raises(SCEncodingError, match="mix encodings"):
        sc_add(a, b)


def test_to_bipolar_from_unipolar() -> None:
    """to_bipolar maps v = 2p - 1, switches encoding, assigns a new corr_id."""
    u = encode(torch.tensor([0.0, 0.25, 1.0]), UNIPOLAR)
    b = to_bipolar(u)
    torch.testing.assert_close(b.value, torch.tensor([-1.0, -0.5, 1.0]), rtol=0.0, atol=0.0)
    assert b.config.encoding == "bipolar"
    assert b.config.length == u.config.length
    assert b.corr_id != u.corr_id
    assert b.scale == u.scale


def test_to_unipolar_from_bipolar() -> None:
    """to_unipolar maps p = (v + 1) / 2, switches encoding, assigns a new corr_id."""
    b = encode(torch.tensor([-1.0, 0.0, 1.0]), BIPOLAR)
    u = to_unipolar(b)
    torch.testing.assert_close(u.value, torch.tensor([0.0, 0.5, 1.0]), rtol=0.0, atol=0.0)
    assert u.config.encoding == "unipolar"
    assert u.config.length == b.config.length
    assert u.corr_id != b.corr_id
    assert u.scale == b.scale


def test_same_encoding_conversion_keeps_identity() -> None:
    """Converting to the encoding a number already has keeps value and corr_id."""
    b = encode(torch.tensor([0.5]), BIPOLAR)
    same_b = to_bipolar(b)
    assert same_b.corr_id == b.corr_id
    torch.testing.assert_close(same_b.value, b.value, rtol=0.0, atol=0.0)
    u = encode(torch.tensor([0.5]), UNIPOLAR)
    same_u = to_unipolar(u)
    assert same_u.corr_id == u.corr_id
    torch.testing.assert_close(same_u.value, u.value, rtol=0.0, atol=0.0)


def test_conversion_round_trip() -> None:
    """unipolar -> bipolar -> unipolar recovers the original values within eps."""
    torch.manual_seed(2)
    x = torch.rand(6)
    u = encode(x, UNIPOLAR)
    back = to_unipolar(to_bipolar(u))
    torch.testing.assert_close(back.value, x)
    assert back.config.encoding == "unipolar"


def test_fresh_corr_ids_distinct_across_encodes() -> None:
    """Every encode gets its own correlation id."""
    x = torch.tensor([0.5])
    ids = [encode(x, BIPOLAR).corr_id for _ in range(5)]
    assert len(set(ids)) == 5
    assert fresh_corr_id() != fresh_corr_id()


def test_decode_descale_division() -> None:
    """decode divides by scale when descale=True and not otherwise."""
    s = SCNumber(torch.tensor([0.5, -0.25]), BIPOLAR, scale=0.25)
    torch.testing.assert_close(decode(s), torch.tensor([2.0, -1.0]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        decode(s, descale=False), torch.tensor([0.5, -0.25]), rtol=0.0, atol=0.0
    )


def test_repr_contains_encoding_and_length() -> None:
    """SCNumber repr names the encoding and the bitstream length N."""
    rb = repr(encode(torch.tensor([0.5]), BIPOLAR))
    assert "bipolar" in rb
    assert "N=256" in rb
    ru = repr(encode(torch.tensor([0.5]), UNIPOLAR))
    assert "unipolar" in ru
    assert "N=256" in ru


@pytest.mark.parametrize("encoding", ["unipolar", "bipolar"])
def test_value_probability_maps_are_inverses(encoding: str) -> None:
    """value_to_probability and probability_to_value invert each other."""
    torch.manual_seed(3)
    p = torch.rand(16)
    torch.testing.assert_close(value_to_probability(probability_to_value(p, encoding), encoding), p)
    v = probability_to_value(torch.rand(16), encoding)
    torch.testing.assert_close(probability_to_value(value_to_probability(v, encoding), encoding), v)


def test_bipolar_probability_map_values() -> None:
    """Bipolar v maps to p = (v + 1) / 2 and unipolar is the identity."""
    v = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(
        value_to_probability(v, "bipolar"), torch.tensor([0.0, 0.5, 1.0]), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(value_to_probability(v, "unipolar"), v, rtol=0.0, atol=0.0)
    s = encode(v, BIPOLAR)
    torch.testing.assert_close(s.probabilities(), torch.tensor([0.0, 0.5, 1.0]), rtol=0.0, atol=0.0)
