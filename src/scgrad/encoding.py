"""SC value encoding: SCConfig, SCNumber, encode/decode, converters.

Unipolar encodes a value x in [0, 1] as P(bit = 1) = x. Bipolar encodes
x in [-1, 1] as P(bit = 1) = (x + 1) / 2. These two maps live here and
nowhere else; every other module imports them.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

Encoding = Literal["unipolar", "bipolar"]
Source = Literal["lfsr", "sobol"]
Accumulator = Literal["mux", "apc"]

_RANGES: dict[str, tuple[float, float]] = {"unipolar": (0.0, 1.0), "bipolar": (-1.0, 1.0)}


class SCEncodingError(Exception):
    """Raised when SC encodings are mixed or a value is not encodable."""


@dataclass(frozen=True)
class SCConfig:
    """Configuration shared by every SCNumber in one circuit.

    length is the bitstream length N: the precision/latency knob (error
    falls as 1/sqrt(N)). source selects the bitstream generator used by
    the exact path. n_rngs models a hardware randomness budget: corr_ids
    are mapped onto n_rngs physical generators (corr_id % n_rngs), so
    streams whose ids collide share randomness and become correlated,
    exactly as they would on silicon with a shared RNG. None means every
    id gets its own generator. noise enables training-time injection of
    the analytic SC counting noise (see layers.py); it has no effect on
    the exact path. accumulator selects how layers sum their products:
    "mux" is the classical scaled-average MUX tree (one selected term
    per clock; counting variance ~ p(1-p)/N regardless of fan-in);
    "apc" is the accumulative parallel counter used by published SC
    accelerators (every product bit counted every clock; variance
    smaller by the fan-in factor, at the hardware cost of a binary
    adder tree). Both produce the same expected value and the same
    scale bookkeeping; they differ in noise, which is the trade the
    library exists to expose.
    """

    encoding: Encoding = "bipolar"
    length: int = 256
    source: Source = "sobol"
    seed: int | None = None
    n_rngs: int | None = None
    noise: bool = False
    accumulator: Accumulator = "mux"

    def rng_index(self, corr_id: int) -> int:
        """Map a correlation id onto a physical generator index."""
        if self.n_rngs is None:
            return corr_id
        return corr_id % self.n_rngs


_id_counter = itertools.count()
_id_lock = threading.Lock()


def fresh_corr_id() -> int:
    """Allocate a new correlation identity for an independent stream."""
    with _id_lock:
        return next(_id_counter)


class _ClampSTE(torch.autograd.Function):
    """Clamp to the encoding range with a straight-through gradient.

    A hard clamp has zero gradient outside the range, which permanently
    stalls any weight that drifts past it during SC-aware training. The
    straight-through estimator passes the gradient unchanged (Bengio et
    al. 2013), the standard remedy from binarized-network training.
    Inside the range the forward is the identity, so gradcheck holds
    there; on the clamped region the pass-through gradient is deliberate
    and not the analytic derivative.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, lo: float, hi: float) -> Tensor:
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor, None, None]:
        return grad_out, None, None


def clamp_ste(x: Tensor, encoding: Encoding) -> Tensor:
    """Clamp x into the valid range of the encoding, gradient passing through."""
    lo, hi = _RANGES[encoding]
    out: Tensor = _ClampSTE.apply(x, lo, hi)  # type: ignore[no-untyped-call]
    return out


class SCNumber:
    """A value in SC representation: probability-space tensor plus metadata.

    value holds the represented value (unipolar in [0,1], bipolar in
    [-1,1]). No bits are stored; the exact path generates them on demand
    from hardware.py using corr_id. scale accumulates MUX-add scale
    factors (ops.py): the raw value is the physical quantity the hardware
    produces, and decode(descale=True) divides the scale back out to
    recover the intended magnitude.
    """

    __slots__ = ("config", "corr_id", "scale", "value")

    def __init__(
        self,
        value: Tensor,
        config: SCConfig,
        scale: float = 1.0,
        corr_id: int | None = None,
    ) -> None:
        self.value = value
        self.config = config
        self.scale = scale
        self.corr_id = fresh_corr_id() if corr_id is None else corr_id

    def __mul__(self, other: SCNumber) -> SCNumber:
        from scgrad.ops import sc_mul

        return sc_mul(self, other)

    def __add__(self, other: SCNumber) -> SCNumber:
        from scgrad.ops import sc_add

        return sc_add(self, other)

    def __repr__(self) -> str:
        v = self.value
        summary = f"shape={tuple(v.shape)}, mean={v.float().mean().item():+.4f}"
        return (
            f"SCNumber({self.config.encoding}, N={self.config.length}, "
            f"{summary}, scale={self.scale:g}, id={self.corr_id})"
        )

    def probabilities(self) -> Tensor:
        """Return P(bit = 1) for this number's stream (the SNG comparator input)."""
        return value_to_probability(self.value, self.config.encoding)


def value_to_probability(value: Tensor, encoding: Encoding) -> Tensor:
    """Map a represented value to its bit probability P(bit = 1)."""
    if encoding == "unipolar":
        return value
    return (value + 1.0) / 2.0


def probability_to_value(p: Tensor, encoding: Encoding) -> Tensor:
    """Map a bit probability back to the represented value."""
    if encoding == "unipolar":
        return p
    return 2.0 * p - 1.0


def encode(x: Tensor, config: SCConfig) -> SCNumber:
    """Encode a float tensor as an SCNumber, clamping into the valid range.

    The clamp uses a straight-through gradient so encoding stays usable
    inside a training graph. A fresh corr_id is assigned: a newly encoded
    value gets its own SNG on hardware.
    """
    if not isinstance(x, Tensor):
        raise SCEncodingError(f"encode expects a Tensor, got {type(x).__name__}")
    return SCNumber(clamp_ste(x, config.encoding), config, scale=1.0)


def decode(s: SCNumber, descale: bool = True) -> Tensor:
    """Return the represented value; with descale, recover the intended magnitude.

    The raw value keeps the accumulated MUX scale because that is the
    physical quantity on the wire. descale divides it back out, which is
    what you want when comparing against a float reference.
    """
    if descale:
        return s.value / s.scale
    return s.value


def to_bipolar(s: SCNumber) -> SCNumber:
    """Convert a unipolar SCNumber to bipolar via v = 2p - 1 (new corr_id)."""
    if s.config.encoding == "bipolar":
        return SCNumber(s.value, s.config, scale=s.scale, corr_id=s.corr_id)
    cfg = SCConfig(
        encoding="bipolar",
        length=s.config.length,
        source=s.config.source,
        seed=s.config.seed,
        n_rngs=s.config.n_rngs,
        noise=s.config.noise,
    )
    return SCNumber(2.0 * s.value - 1.0, cfg, scale=s.scale)


def to_unipolar(s: SCNumber) -> SCNumber:
    """Convert a bipolar SCNumber to unipolar via p = (v + 1) / 2 (new corr_id).

    Note the affine map does not commute with MUX scaling: converting a
    scaled bipolar value reinterprets the same physical stream, so the
    scale is carried over unchanged and decode(descale=True) is only
    meaningful for scale = 1 after conversion.
    """
    if s.config.encoding == "unipolar":
        return SCNumber(s.value, s.config, scale=s.scale, corr_id=s.corr_id)
    cfg = SCConfig(
        encoding="unipolar",
        length=s.config.length,
        source=s.config.source,
        seed=s.config.seed,
        n_rngs=s.config.n_rngs,
        noise=s.config.noise,
    )
    return SCNumber((s.value + 1.0) / 2.0, cfg, scale=s.scale)


def require_same_encoding(a: SCNumber, b: SCNumber) -> None:
    """Raise SCEncodingError when two SCNumbers mix encodings."""
    if a.config.encoding != b.config.encoding:
        raise SCEncodingError(f"cannot mix encodings: {a.config.encoding} vs {b.config.encoding}")
