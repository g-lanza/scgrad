"""Tests for scgrad.accuracy: the analytic error model against physical bitstreams.

The analytic claims under test: counting noise falls as 1/sqrt(N), the
uGEMM scaled-add bounds are reproduced exactly, the estimator brackets
the empirical exact-path counting error within a small factor, and
calibration moves predictions toward measured points in log space.
"""

from __future__ import annotations

import math
from functools import cache
from itertools import pairwise

import pytest
import torch

from scgrad.accuracy import (
    accuracy_estimator,
    counting_std,
    sc_noise_std,
    scaled_add_max_error,
    tolerance,
)
from scgrad.encoding import SCConfig
from scgrad.hardware import make_source

LENGTHS = [64, 256, 1024, 4096]
VALUES = [-0.62, -0.21, 0.137, 0.5, 0.731]
CORR_IDS = [3, 11, 27, 64]

SMALL_FACTOR = 10.0
LARGE_FACTOR = 200.0


@cache
def _measured_counting_errors(source_name: str) -> dict[int, float]:
    """Mean |counted value - true value| from real bitstreams at each length.

    Bits come straight from the hardware source; the counted bipolar
    value is 2 * mean(bits) - 1. Errors are averaged over several values
    and correlation ids. Fully deterministic given the config seed.
    """
    torch.manual_seed(0)
    config = SCConfig(encoding="bipolar", length=256, source=source_name, seed=7)
    source = make_source(config)
    out: dict[int, float] = {}
    for n in LENGTHS:
        errors = []
        for corr_id in CORR_IDS:
            for v in VALUES:
                p = torch.tensor([(v + 1.0) / 2.0], dtype=torch.float64)
                bits = source.bits(p, n, corr_id)
                counted = 2.0 * bits.double().mean().item() - 1.0
                errors.append(abs(counted - v))
        out[n] = sum(errors) / len(errors)
    return out


def _log_residual(estimator_error: float, measured: float) -> float:
    return math.log(measured / estimator_error) ** 2


class TestCountingStd:
    def test_halves_when_length_quadruples(self) -> None:
        for encoding in ("unipolar", "bipolar"):
            for n in (64, 256, 1024):
                assert counting_std(4 * n, encoding) == pytest.approx(
                    counting_std(n, encoding) / 2.0
                )

    def test_matches_worst_case_formula(self) -> None:
        assert counting_std(256, "unipolar") == pytest.approx(0.5 / 16.0)
        assert counting_std(256, "bipolar") == pytest.approx(1.0 / 16.0)
        assert counting_std(1024, "bipolar") == pytest.approx(2.0 * counting_std(1024, "unipolar"))


class TestTolerance:
    def test_decreases_with_length(self) -> None:
        for encoding in ("unipolar", "bipolar"):
            tols = [tolerance(n, depth=1, encoding=encoding) for n in LENGTHS]
            assert all(later < earlier for earlier, later in pairwise(tols))

    def test_increases_with_depth(self) -> None:
        for encoding in ("unipolar", "bipolar"):
            tols = [tolerance(256, depth=d, encoding=encoding) for d in (1, 2, 4, 8)]
            assert all(later > earlier for earlier, later in pairwise(tols))

    def test_bipolar_wider_than_unipolar(self) -> None:
        assert tolerance(256, 1, "bipolar") > tolerance(256, 1, "unipolar")


class TestScaledAddMaxError:
    @pytest.mark.parametrize("n_terms", [2, 8, 64])
    @pytest.mark.parametrize("length", [256, 4096])
    def test_matches_ugemm_formulas(self, n_terms: int, length: int) -> None:
        unipolar_bound = (n_terms - 1) / (length * n_terms)
        assert scaled_add_max_error(n_terms, length, "unipolar") == pytest.approx(unipolar_bound)
        assert scaled_add_max_error(n_terms, length, "bipolar") == pytest.approx(
            2.0 * unipolar_bound
        )


class TestEstimatorAnalytic:
    def test_composes_counting_and_add_terms(self) -> None:
        est = accuracy_estimator(4, ["mul", "add:64", "mul", "add:8"], "bipolar")
        expected = (
            math.sqrt(4) * counting_std(256, "bipolar")
            + scaled_add_max_error(64, 256, "bipolar")
            + scaled_add_max_error(8, 256, "bipolar")
        )
        assert est.error(256) == pytest.approx(expected)

    def test_mul_stages_add_no_extra_error(self) -> None:
        with_muls = accuracy_estimator(2, ["mul", "mul"], "bipolar")
        without = accuracy_estimator(2, [], "bipolar")
        assert with_muls.error(1024) == pytest.approx(without.error(1024))

    def test_curve_is_decreasing_over_default_grid(self) -> None:
        est = accuracy_estimator(1, ["add:16"], "bipolar")
        curve = est.curve()
        assert len(curve) > 1
        errors = [e for _, e in curve]
        assert all(later < earlier for earlier, later in pairwise(errors))


class TestEmpiricalAgainstEstimate:
    @pytest.mark.parametrize("source_name", ["sobol", "lfsr"])
    def test_measured_error_within_factor_of_analytic(self, source_name: str) -> None:
        measured = _measured_counting_errors(source_name)
        est = accuracy_estimator(1, [], "bipolar")
        for n in LENGTHS:
            predicted = est.error(n)
            assert measured[n] <= predicted * SMALL_FACTOR
            assert measured[n] >= predicted / LARGE_FACTOR

    def test_sobol_error_shows_inverse_sqrt_trend(self) -> None:
        measured = _measured_counting_errors("sobol")
        assert measured[4096] < 0.5 * measured[256]
        assert measured[4096] < measured[64]

    def test_lfsr_error_decreases_with_length(self) -> None:
        measured = _measured_counting_errors("lfsr")
        assert measured[4096] < measured[64]


class TestCalibrate:
    def test_calibration_on_measured_points_reduces_log_residuals(self) -> None:
        measured = _measured_counting_errors("sobol")
        points = [(n, measured[n]) for n in LENGTHS]
        est = accuracy_estimator(1, [], "bipolar")
        pre = sum(_log_residual(est.error(n), err) for n, err in points)
        correction = est.calibrate(points)
        post = sum(_log_residual(est.error(n), err) for n, err in points)
        assert post < pre
        assert 0.0 < correction < 1.0

    def test_correction_is_geometric_mean_of_ratios(self) -> None:
        est = accuracy_estimator(1, ["add:8"], "bipolar")
        ratios = [0.35, 0.25, 0.30]
        lengths = [64, 256, 1024]
        points = [(n, r * est.error(n)) for n, r in zip(lengths, ratios, strict=True)]
        correction = est.calibrate(points)
        geo_mean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        assert correction == pytest.approx(geo_mean)
        post = sum(_log_residual(est.error(n), err) for n, err in points)
        uncalibrated = accuracy_estimator(1, ["add:8"], "bipolar")
        pre = sum(_log_residual(uncalibrated.error(n), err) for n, err in points)
        assert post < pre

    def test_calibrate_skips_zero_measurements(self) -> None:
        est = accuracy_estimator(1, [], "bipolar")
        correction = est.calibrate([(256, 0.0)])
        assert correction == pytest.approx(1.0)


class TestScNoiseStd:
    def test_unipolar_formula_at_sample_points(self) -> None:
        values = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)
        out = sc_noise_std(values, 256, "unipolar")
        expected = torch.sqrt(values * (1.0 - values) / 256.0)
        assert torch.allclose(out, expected)
        assert out[2].item() == pytest.approx(1.0 / 32.0)
        assert out[0].item() == 0.0
        assert out[3].item() == 0.0

    def test_bipolar_formula_at_sample_points(self) -> None:
        values = torch.tensor([-1.0, -0.5, 0.0, 0.6, 1.0], dtype=torch.float64)
        out = sc_noise_std(values, 256, "bipolar")
        expected = torch.sqrt((1.0 - values * values) / 256.0)
        assert torch.allclose(out, expected)
        assert out[2].item() == pytest.approx(1.0 / 16.0)
        assert out[0].item() == 0.0
        assert out[4].item() == 0.0

    def test_clamps_negative_variance_to_zero(self) -> None:
        out = sc_noise_std(torch.tensor([1.2], dtype=torch.float64), 64, "bipolar")
        assert out.item() == 0.0
        assert not torch.isnan(out).any()
