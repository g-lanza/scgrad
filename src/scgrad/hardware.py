"""Bitstream generation for the exact path: LFSR and Sobol sources.

This is the physical-truth generator. All randomness in the exact path
flows through here; no other module calls torch.rand for bit generation.
Streams are deterministic given (seed, corr_id). Distinct correlation ids
map to distinct generator states and produce statistically independent
streams; ids that collide under SCConfig.n_rngs share a generator and
produce maximally correlated streams, as on hardware with a shared RNG.

Elements within one tensor share the random number sequence (one RNS
driving many comparator SNGs), which is the standard hardware layout.
Cross-tensor independence is what multiplication correctness needs, and
that is governed by corr_id.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from scgrad.encoding import SCConfig

LFSR_WIDTH = 16
_LFSR_PERIOD = (1 << LFSR_WIDTH) - 1
# Fibonacci LFSR with taps (16, 15, 13, 4): maximal length, period 2^16 - 1.
_LFSR_TAPS = (15, 14, 12, 3)


class BitstreamSource(Protocol):
    """A deterministic generator of SC bitstreams."""

    def bits(self, p: Tensor, n: int, corr_id: int) -> Tensor:
        """Return a (*p.shape, n) bool tensor; each bit is 1 with probability p."""
        ...

    def uniforms(self, n: int, corr_id: int) -> Tensor:
        """Return the (n,) random-number sequence in [0, 1) driving this id's SNG.

        bit_t = (uniforms[t] < p) for every element sharing the id; the
        exact path uses this to gather selected bits without
        materializing whole streams.
        """
        ...


def _lfsr_cycle() -> Tensor:
    """Generate the full 2^16 - 1 state cycle of the maximal 16-bit LFSR."""
    states = torch.empty(_LFSR_PERIOD, dtype=torch.int64)
    s = 0xACE1
    for i in range(_LFSR_PERIOD):
        states[i] = s
        fb = 0
        for t in _LFSR_TAPS:
            fb ^= s >> t
        s = ((s << 1) | (fb & 1)) & ((1 << LFSR_WIDTH) - 1)
    return states


_CYCLE_CACHE: Tensor | None = None


def _cycle() -> Tensor:
    global _CYCLE_CACHE
    if _CYCLE_CACHE is None:
        _CYCLE_CACHE = _lfsr_cycle()
    return _CYCLE_CACHE


def _comparator(seq01: Tensor, p: Tensor) -> Tensor:
    """Comparator SNG: bit_t = (r_t < p), vectorized over the value tensor.

    seq01 is the (n,) random-number sequence in [0, 1); p is the value
    tensor of bit probabilities. Output shape (*p.shape, n).
    """
    flat = p.reshape(-1, 1)
    bits = seq01.reshape(1, -1) < flat
    return bits.reshape(*p.shape, -1)


class LFSRSource:
    """Maximal-length 16-bit LFSR comparator SNG.

    Distinct corr_ids are distinct phases of the m-sequence; m-sequence
    cross-correlation at nonzero lag is -1/period, i.e. effectively zero.
    For n beyond the 2^16 - 1 period the sequence repeats, exactly as a
    16-bit hardware LFSR would.
    """

    def __init__(self, config: SCConfig) -> None:
        self.config = config
        self.seed = config.seed if config.seed is not None else 0

    def uniforms(self, n: int, corr_id: int) -> Tensor:
        rng = self.config.rng_index(corr_id)
        offset = (self.seed * 31 + rng * 9973) % _LFSR_PERIOD
        idx = (offset + torch.arange(n)) % _LFSR_PERIOD
        # States span 1..2^16-1; map to [0, 1) so the comparator sees uniforms.
        return (_cycle()[idx].to(torch.float64) - 1.0) / float(_LFSR_PERIOD)

    def bits(self, p: Tensor, n: int, corr_id: int) -> Tensor:
        return _comparator(self.uniforms(n, corr_id), p.to(torch.float64))


_SOBOL_MAX_DIMS = 1024


class SobolSource:
    """Sobol low-discrepancy SNG: one Sobol dimension per generator index.

    Low-discrepancy streams converge much faster than LFSR pseudo-random
    streams at equal length (Najafi/Liu line of work; Hsiao et al. in
    Gross & Gaudet 2019). Joint uniformity matters as much as marginal
    uniformity: AND/XNOR multiplication needs the pair (r_a(t), r_b(t))
    to fill the unit square, which two separately constructed
    one-dimensional sequences do not do. Sobol dimensions are designed
    to be jointly equidistributed, so each generator index maps to its
    own dimension of one sequence. The seed fast-forwards the sequence
    (skipping the all-zeros first point); generator indices wrap at
    1024 dimensions, a finite-hardware budget documented here.
    """

    def __init__(self, config: SCConfig) -> None:
        self.config = config
        self.seed = config.seed if config.seed is not None else 0

    def uniforms(self, n: int, corr_id: int) -> Tensor:
        dim = self.config.rng_index(corr_id) % _SOBOL_MAX_DIMS
        engine = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
            dimension=dim + 1, scramble=False
        )
        engine.fast_forward(1 + self.seed * 64)
        return engine.draw(n, dtype=torch.float64)[:, dim]

    def bits(self, p: Tensor, n: int, corr_id: int) -> Tensor:
        return _comparator(self.uniforms(n, corr_id), p.to(torch.float64))


def make_source(config: SCConfig) -> BitstreamSource:
    """Build the bitstream source selected by the config."""
    if config.source == "lfsr":
        return LFSRSource(config)
    return SobolSource(config)


def stream_for(value: Tensor, config: SCConfig, corr_id: int, n: int | None = None) -> Tensor:
    """Generate the real bitstream for a value tensor (helper for the exact path)."""
    from scgrad.encoding import value_to_probability

    source = make_source(config)
    p = value_to_probability(value.detach(), config.encoding)
    return source.bits(p, n if n is not None else config.length, corr_id)
