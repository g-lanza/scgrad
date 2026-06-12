"""Dual-path invariant for the APC accumulator (the accelerator-style adder).

APC counts every product bit every clock, so its counting noise is the
MUX noise divided by sqrt(fan_in). The expected value and the scale
bookkeeping are identical to MUX; only the noise differs. Tolerances
come from accuracy.apc_counting_std, never tuned to pass.
"""

import torch
from torch import nn

from scgrad.accuracy import _COMPARATOR_QUANT, apc_counting_std
from scgrad.encoding import SCConfig, SCNumber
from scgrad.eval_exact import exact_forward
from scgrad.layers import SCLinear, SCReLU


def _apc_tolerance(n: int, k: int, depth: int = 1) -> float:
    return 5.0 * depth**0.5 * apc_counting_std(k, n) + depth * _COMPARATOR_QUANT


def _dual_path_error(model: nn.Sequential, x: torch.Tensor, cfg: SCConfig) -> float:
    model.eval()
    with torch.no_grad():
        approx = model(x)
    assert isinstance(approx, SCNumber)
    exact = exact_forward(model, x, cfg)
    assert approx.scale == exact.scale
    return float((approx.value - exact.value).abs().mean().item())


def test_apc_linear_converges() -> None:
    errors = {}
    for n in (64, 256, 1024, 4096):
        cfg = SCConfig(length=n, seed=3, accumulator="apc")
        torch.manual_seed(0)
        model = nn.Sequential(SCLinear(8, 6, config=cfg))
        x = torch.rand(5, 8) * 2 - 1
        err = _dual_path_error(model, x, cfg)
        assert err <= _apc_tolerance(n, 9), (n, err, _apc_tolerance(n, 9))
        errors[n] = err
    assert errors[4096] < 0.5 * errors[64] + 1e-4, errors


def test_apc_mlp_converges() -> None:
    errors = {}
    for n in (256, 1024, 4096):
        cfg = SCConfig(length=n, seed=3, accumulator="apc")
        torch.manual_seed(0)
        model = nn.Sequential(SCLinear(8, 6, config=cfg), SCReLU(), SCLinear(6, 4, config=cfg))
        x = torch.rand(5, 8) * 2 - 1
        err = _dual_path_error(model, x, cfg)
        assert err <= _apc_tolerance(n, 7, depth=2), (n, err)
        errors[n] = err
    assert errors[4096] < 0.7 * errors[256] + 1e-4, errors


def test_apc_noise_smaller_than_mux() -> None:
    """Training-noise injection reflects the APC variance advantage."""
    torch.manual_seed(0)
    x = torch.rand(64, 32) * 2 - 1
    spreads = {}
    for acc in ("mux", "apc"):
        cfg = SCConfig(length=256, seed=3, noise=True, accumulator=acc)
        torch.manual_seed(1)
        layer = SCLinear(32, 16, config=cfg)
        layer.train()
        outs = torch.stack([layer(x).value for _ in range(8)])
        spreads[acc] = float(outs.std(dim=0).mean().item())
    assert spreads["apc"] < spreads["mux"] / 2.0, spreads


def test_apc_expected_value_matches_mux() -> None:
    """Same circuit, same expectation: only the noise model differs."""
    torch.manual_seed(0)
    x = torch.rand(4, 8) * 2 - 1
    values = {}
    for acc in ("mux", "apc"):
        cfg = SCConfig(length=256, seed=3, accumulator=acc)
        torch.manual_seed(2)
        layer = SCLinear(8, 5, config=cfg)
        layer.eval()
        values[acc] = layer(x).value
    torch.testing.assert_close(values["mux"], values["apc"], rtol=0.0, atol=0.0)
