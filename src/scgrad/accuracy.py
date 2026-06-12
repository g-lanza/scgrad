"""Predict SC error from bitstream length, depth, and circuit structure.

The analytic core: counting a Bernoulli stream of length L estimates its
probability with variance p(1 - p)/L, i.e. a standard deviation of at
most 0.5/sqrt(L) in probability space (1/sqrt(L) for bipolar values,
whose range is twice as wide). Each layer of a circuit re-counts a
stream, so independent stage errors compose as sqrt(depth). Scaled MUX
addition contributes the uGEMM worst-case bounds: (k - 1)/(L * k) for a
k-term unipolar add and twice that for bipolar.

This module also defines tolerance(), the bound tests/test_dual_path.py
holds the approximate path to. The tolerance is derived from the
analytic model, never tuned to make a failing comparison pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from scgrad.encoding import Encoding

_COMPARATOR_QUANT = 2.0**-16


def counting_std(length: int, encoding: Encoding = "bipolar") -> float:
    """Worst-case standard deviation of a counted stream estimate of length L."""
    base = 0.5 / math.sqrt(length)
    return 2.0 * base if encoding == "bipolar" else base


def scaled_add_max_error(n_terms: int, length: int, encoding: Encoding = "bipolar") -> float:
    """uGEMM worst-case error bound for a k-term scaled MUX addition."""
    bound = (n_terms - 1) / (length * n_terms)
    return 2.0 * bound if encoding == "bipolar" else bound


def tolerance(n: int, depth: int = 1, encoding: Encoding = "bipolar") -> float:
    """Dual-path tolerance: bound on mean |exact - approx| at stream length n.

    Five standard deviations of the composed counting noise plus the
    comparator quantization floor. Mean absolute error of zero-mean noise
    sits near 0.8 standard deviations, so the factor of five is a slack
    bound that still fails loudly on systematic bugs (a forgotten scale
    factor is an O(1) discrepancy).
    """
    noise = 5.0 * math.sqrt(depth) * counting_std(n, encoding)
    return noise + depth * _COMPARATOR_QUANT


@dataclass
class Estimator:
    """Analytic SC error model for a circuit, with optional empirical correction.

    op_types lists the circuit's stages as "mul" or "add:<k>" entries
    (k = MUX fan-in). error(length) returns the predicted absolute error
    of the circuit output at the given stream length; calibrate() fits a
    multiplicative correction against measured exact-path error so the
    estimate matches a particular model and dataset.
    """

    circuit_depth: int
    op_types: list[str]
    encoding: Encoding = "bipolar"
    correction: float = 1.0
    calibration: list[tuple[int, float]] = field(default_factory=list)

    def _analytic(self, length: int) -> float:
        err = math.sqrt(self.circuit_depth) * counting_std(length, self.encoding)
        for op in self.op_types:
            if op.startswith("add:"):
                err += scaled_add_max_error(int(op.split(":", 1)[1]), length, self.encoding)
        return err

    def error(self, length: int) -> float:
        """Predicted absolute output error at stream length `length`."""
        return self.correction * self._analytic(length)

    def curve(self, lengths: list[int] | None = None) -> list[tuple[int, float]]:
        """(length, error) pairs across a default or given grid of lengths."""
        grid = lengths if lengths is not None else [2**k for k in range(4, 15)]
        return [(n, self.error(n)) for n in grid]

    def calibrate(self, measured: list[tuple[int, float]]) -> float:
        """Fit the multiplicative correction to measured (length, error) points.

        Least squares in log space: the correction is the geometric mean
        of measured/analytic ratios. Returns the fitted correction.
        Measurements come from exact-path runs (see eval_exact and
        tests/test_accuracy.py); zero measurements are skipped.
        """
        ratios = [math.log(err / self._analytic(n)) for n, err in measured if err > 0.0]
        if ratios:
            self.correction = math.exp(sum(ratios) / len(ratios))
        self.calibration = list(measured)
        return self.correction

    def plot(self, lengths: list[int] | None = None) -> str:
        """Render the error curve log-log with plotext (requires scgrad[gui])."""
        try:
            import plotext
        except ImportError as exc:
            raise ImportError("Estimator.plot requires plotext: install scgrad[gui]") from exc
        points = self.curve(lengths)
        plotext.clear_figure()
        plotext.plot(
            [math.log2(n) for n, _ in points],
            [math.log10(e) for _, e in points],
            marker="braille",
        )
        plotext.title("SC error vs stream length")
        plotext.xlabel("log2 N")
        plotext.ylabel("log10 error")
        result: str = plotext.build()
        return result


def accuracy_estimator(
    circuit_depth: int, op_types: list[str], encoding: Encoding = "bipolar"
) -> Estimator:
    """Build the analytic error estimator for a circuit (see Estimator)."""
    return Estimator(circuit_depth=circuit_depth, op_types=list(op_types), encoding=encoding)


def apc_counting_std(n_terms: int, length: int, encoding: Encoding = "bipolar") -> float:
    """Worst-case std of an APC-accumulated k-term average at stream length L.

    Each of the k product streams is counted exactly every clock, so the
    per-term variances average down: var <= (1/k^2) * k * worst_per_term
    = worst_per_term / k, i.e. the MUX counting std divided by sqrt(k).
    """
    return counting_std(length, encoding) / math.sqrt(n_terms)


def sc_noise_std(value: Tensor, length: int, encoding: Encoding) -> Tensor:
    """Per-element counting-noise standard deviation of a materialized stream.

    For a stream whose represented value is v: unipolar variance is
    v(1 - v)/L; bipolar variance is (1 - v^2)/L. This is the noise
    layers inject during SC-aware training so the optimizer feels the
    finite stream length (see layers.py).
    """
    if encoding == "unipolar":
        var = value * (1.0 - value) / length
    else:
        var = (1.0 - value * value) / length
    return torch.sqrt(torch.clamp(var, min=0.0))
