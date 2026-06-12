"""Tests for scgrad.hardware: bitstream statistics, reproducibility, independence.

Covers both sources (sobol, lfsr): empirical P(bit=1), determinism under
seed+corr_id, cross-stream independence via SCC, n_rngs collision producing
maximal correlation, XNOR multiply sanity in bipolar, Sobol-vs-LFSR
convergence, and the uniforms() range contract. All streams are fully
deterministic given (seed, corr_id), so every assertion here is exact-repeatable.
"""

from __future__ import annotations

import math

import pytest
import torch

from scgrad.correlation import scc
from scgrad.encoding import SCConfig
from scgrad.hardware import make_source

N = 4096
SOURCES = ["sobol", "lfsr"]

torch.manual_seed(0)


def _config(source: str, seed: int = 7, n_rngs: int | None = None) -> SCConfig:
    return SCConfig(encoding="bipolar", length=N, source=source, seed=seed, n_rngs=n_rngs)


@pytest.mark.parametrize("source", SOURCES)
def test_empirical_probability_matches_p(source: str) -> None:
    """Empirical bit frequency tracks p within 3/sqrt(n) across a grid of p values."""
    n = 2048
    src = make_source(_config(source))
    grid = torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], dtype=torch.float64)
    bits = src.bits(grid, n, corr_id=0)
    assert bits.shape == (grid.shape[0], n)
    assert bits.dtype == torch.bool
    emp = bits.to(torch.float64).mean(dim=-1)
    tol = 3.0 / math.sqrt(n)
    assert (emp - grid).abs().max().item() <= tol


@pytest.mark.parametrize("source", SOURCES)
def test_degenerate_probabilities(source: str) -> None:
    """p=0 yields all-zero streams and p=1 all-one streams (uniforms live in [0,1))."""
    src = make_source(_config(source))
    p = torch.tensor([0.0, 1.0], dtype=torch.float64)
    bits = src.bits(p, 512, corr_id=0)
    assert not bits[0].any()
    assert bits[1].all()


@pytest.mark.parametrize("source", SOURCES)
def test_reproducibility_same_seed_and_id(source: str) -> None:
    """Same seed + corr_id gives identical bits, from the same and a fresh source."""
    cfg = _config(source)
    p = torch.tensor([0.3, 0.5, 0.8], dtype=torch.float64)
    src = make_source(cfg)
    first = src.bits(p, N, corr_id=3)
    second = src.bits(p, N, corr_id=3)
    fresh = make_source(cfg).bits(p, N, corr_id=3)
    assert torch.equal(first, second)
    assert torch.equal(first, fresh)


@pytest.mark.parametrize("source", SOURCES)
def test_different_seed_changes_stream(source: str) -> None:
    """A different config seed shifts the underlying sequence for the same corr_id.

    The check uses uniforms plus bits at p=0.3: Sobol bits at exactly
    p=0.5 only see the MSB of the dim-0 points, which is invariant under
    the even fast-forward offset the seed applies, so p=0.5 is not a
    valid probe of the seed contract.
    """
    p = torch.tensor(0.3, dtype=torch.float64)
    src_a = make_source(_config(source, seed=7))
    src_b = make_source(_config(source, seed=8))
    assert not torch.equal(src_a.uniforms(N, 0), src_b.uniforms(N, 0))
    assert not torch.equal(src_a.bits(p, N, corr_id=0), src_b.bits(p, N, corr_id=0))


@pytest.mark.parametrize("source", SOURCES)
def test_distinct_corr_ids_are_independent(source: str) -> None:
    """Distinct corr_ids give near-zero SCC; Sobol pairs are low-discrepancy-tight."""
    src = make_source(_config(source))
    p = torch.tensor(0.5, dtype=torch.float64)
    threshold = 0.02 if source == "sobol" else 0.15
    for id_a, id_b in [(0, 1), (1, 2), (0, 5), (3, 4)]:
        a = src.bits(p, N, id_a)
        b = src.bits(p, N, id_b)
        assert scc(a, b).abs().item() < threshold


@pytest.mark.parametrize("source", SOURCES)
def test_n_rngs_collision_gives_maximal_correlation(source: str) -> None:
    """With n_rngs=1 two distinct ids share the generator: equal p gives SCC = +1."""
    src = make_source(_config(source, n_rngs=1))
    p = torch.tensor(0.5, dtype=torch.float64)
    a = src.bits(p, N, corr_id=0)
    b = src.bits(p, N, corr_id=1)
    assert torch.equal(a, b)
    assert scc(a, b).item() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("source", SOURCES)
def test_xnor_decodes_to_bipolar_product(source: str) -> None:
    """XNOR of independent bipolar streams for 0.6 and -0.4 decodes to about -0.24."""
    src = make_source(_config(source))
    v_a, v_b = 0.6, -0.4
    p_a = torch.tensor((v_a + 1.0) / 2.0, dtype=torch.float64)
    p_b = torch.tensor((v_b + 1.0) / 2.0, dtype=torch.float64)
    bits_a = src.bits(p_a, N, corr_id=0)
    bits_b = src.bits(p_b, N, corr_id=1)
    xnor = bits_a == bits_b
    decoded = 2.0 * xnor.to(torch.float64).mean().item() - 1.0
    assert decoded == pytest.approx(v_a * v_b, abs=0.06)


def test_sobol_converges_faster_than_lfsr() -> None:
    """At n=256 the mean single-stream estimate error is smaller for Sobol than LFSR."""
    n = 256
    grid = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        dtype=torch.float64,
    )
    errors: dict[str, float] = {}
    for source in SOURCES:
        src = make_source(_config(source))
        per_id = []
        for corr_id in (0, 1, 2):
            emp = src.bits(grid, n, corr_id).to(torch.float64).mean(dim=-1)
            per_id.append((emp - grid).abs().mean().item())
        errors[source] = sum(per_id) / len(per_id)
    assert errors["sobol"] < errors["lfsr"]


@pytest.mark.parametrize("source", SOURCES)
def test_uniforms_in_unit_interval(source: str) -> None:
    """uniforms(n, corr_id) returns an (n,) float64 sequence inside [0, 1)."""
    src = make_source(_config(source))
    for corr_id in (0, 1, 7):
        u = src.uniforms(N, corr_id)
        assert u.shape == (N,)
        assert u.dtype == torch.float64
        assert u.min().item() >= 0.0
        assert u.max().item() < 1.0
