"""Correlation: measured exactly on bits, penalized differentiably in training.

Independence between the two streams entering a multiply is the
precondition for SC multiplication to compute a product at all; with a
shared randomness source the AND of two comparator SNG streams computes
min(p_a, p_b), not p_a * p_b. This module provides (a) the Alaghi-Hayes
SCC metric on real bitstreams for the exact path, (b) a tracker that
records which streams meet at multiplies during a forward pass, and (c) a
differentiable penalty for the correlation-induced multiply error under
the configured hardware randomness budget (SCConfig.n_rngs).

The penalty is not the SCC itself (sampled bits are not differentiable).
For comparator SNGs sharing one RNS the correlated multiply output has a
closed form: unipolar AND gives min(p_a, p_b); bipolar XNOR gives
v = 1 - 2|p_a - p_b|. The penalty is the mean absolute gap between that
correlated output and the intended independent product. It is exactly the
hardware error the collision would cause, it is differentiable almost
everywhere in the model parameters, and it is zero when no streams
collide. Validated against exact-path measurements in
tests/test_correlation.py. Known limitation: it models the maximal-
correlation (SCC = +1) collision case, which is what a shared comparator
RNS produces; partial correlation from upstream op outputs is not
modeled in v0.1 (see docs/design_notes.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from scgrad.encoding import Encoding, SCConfig, SCNumber


def scc(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """Alaghi-Hayes stochastic cross-correlation in [-1, 1] on real bitstreams.

    Inputs are (*shape, n) bit tensors; the result has shape (*shape,).
    SCC = +1 for maximally positively correlated streams, 0 for
    independent, -1 for maximally negatively correlated. Degenerate
    streams (all zeros or all ones) have no correlation freedom; their
    SCC is defined as 0.
    """
    a = a_bits.to(torch.float64)
    b = b_bits.to(torch.float64)
    p1 = a.mean(dim=-1)
    p2 = b.mean(dim=-1)
    p11 = (a * b).mean(dim=-1)
    delta = p11 - p1 * p2
    pos_den = torch.minimum(p1, p2) - p1 * p2
    neg_den = p1 * p2 - torch.clamp(p1 + p2 - 1.0, min=0.0)
    den = torch.where(delta > 0, pos_den, neg_den)
    safe = den.abs() > 1e-12
    ratio = delta / torch.where(safe, den, torch.ones_like(den))
    return torch.where(safe, ratio, torch.zeros_like(delta))


@dataclass
class MultiplyEvent:
    """One multiply observed during a tracked forward pass."""

    corr_id_a: int
    corr_id_b: int
    p_a: Tensor
    p_b: Tensor
    encoding: Encoding
    config: SCConfig


@dataclass
class CorrelationTracker:
    """Records the streams that meet at each multiply during a forward pass.

    Use as a context manager around a forward pass, then hand it to
    correlation_loss. Recording keeps the probability tensors with their
    autograd graph so the penalty is differentiable in the parameters.
    """

    events: list[MultiplyEvent] = field(default_factory=list)

    def __enter__(self) -> CorrelationTracker:
        _ACTIVE.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        _ACTIVE.remove(self)

    def record(self, event: MultiplyEvent) -> None:
        self.events.append(event)

    def collisions(self) -> list[MultiplyEvent]:
        """Events whose two streams share a physical generator."""
        return [
            e
            for e in self.events
            if e.config.rng_index(e.corr_id_a) == e.config.rng_index(e.corr_id_b)
        ]


_ACTIVE: list[CorrelationTracker] = []


def record_multiply(a: SCNumber, b: SCNumber) -> None:
    """Record a multiply with the active tracker, if any (called by ops)."""
    if not _ACTIVE:
        return
    event = MultiplyEvent(
        corr_id_a=a.corr_id,
        corr_id_b=b.corr_id,
        p_a=a.probabilities(),
        p_b=b.probabilities(),
        encoding=a.config.encoding,
        config=a.config,
    )
    for tracker in _ACTIVE:
        tracker.record(event)


def correlated_multiply_error(event: MultiplyEvent) -> Tensor:
    """Differentiable per-element error a shared-RNS collision causes at this multiply.

    Closed forms for SCC = +1 comparator sharing: unipolar AND output is
    min(p_a, p_b) against the intended p_a * p_b; bipolar XNOR output is
    1 - 2|p_a - p_b| against the intended v_a * v_b.
    """
    pa, pb = event.p_a, event.p_b
    if event.encoding == "unipolar":
        return (torch.minimum(pa, pb) - pa * pb).abs()
    va = 2.0 * pa - 1.0
    vb = 2.0 * pb - 1.0
    correlated = 1.0 - 2.0 * (pa - pb).abs()
    return (correlated - va * vb).abs()


def correlation_loss(tracker: CorrelationTracker) -> Tensor:
    """Differentiable penalty: total correlation-induced multiply error.

    Sums the mean per-element collision error over every multiply whose
    streams share a generator under the configured randomness budget.
    Returns a connected scalar; zero when nothing collides. Minimizing it
    drives the optimizer toward parameter configurations whose values
    suffer least when forced through correlated streams.
    """
    collisions = tracker.collisions()
    if not collisions:
        return torch.zeros(())
    total = torch.zeros(())
    for event in collisions:
        total = total + correlated_multiply_error(event).mean()
    return total
