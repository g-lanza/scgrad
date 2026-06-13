"""Dual-path agreement for a conv->flatten->linear pipeline.

Locks the SCFlatten wiring: an SCConv2d feeding an SCLinear through
SCFlatten must agree between the approximate and exact paths, with scale
carried correctly across the reshape. This is the shape the CIFAR
benchmark uses.
"""

import torch
from torch import nn

from scgrad.accuracy import apc_counting_std
from scgrad.encoding import SCConfig, SCNumber
from scgrad.eval_exact import exact_forward
from scgrad.layers import SCConv2d, SCFlatten, SCLinear, SCReLU


def _apc_tol(n: int, k: int, depth: int) -> float:
    return 5.0 * depth**0.5 * apc_counting_std(k, n) + depth * 2.0**-16


def test_conv_flatten_linear_dual_path() -> None:
    errors = {}
    for n in (256, 1024, 4096):
        cfg = SCConfig(length=n, seed=4, accumulator="apc")
        torch.manual_seed(0)
        model = nn.Sequential(
            SCConv2d(1, 3, 3, stride=2, config=cfg),
            SCReLU(),
            SCFlatten(),
            SCLinear(3 * 3 * 3, 4, config=cfg),
        )
        x = torch.rand(4, 1, 8, 8) * 2 - 1
        model.eval()
        with torch.no_grad():
            approx = model(x)
        assert isinstance(approx, SCNumber)
        exact = exact_forward(model, x, cfg)
        assert approx.scale == exact.scale
        err = float((approx.value - exact.value).abs().mean().item())
        conv_k = 1 * 3 * 3 + 1
        lin_k = 3 * 3 * 3 + 1
        assert err <= _apc_tol(n, max(conv_k, lin_k), depth=2), (n, err)
        errors[n] = err
    assert errors[4096] < 0.7 * errors[256] + 1e-4, errors


def test_scflatten_preserves_metadata() -> None:
    cfg = SCConfig(length=256, seed=4)
    torch.manual_seed(0)
    conv = SCConv2d(2, 3, 3, config=cfg)
    x = torch.rand(5, 2, 6, 6) * 2 - 1
    conv.eval()
    out = conv(x)
    assert isinstance(out, SCNumber)
    flat = SCFlatten()(out)
    assert flat.scale == out.scale
    assert flat.corr_id == out.corr_id
    assert flat.value.shape == (5, out.value[0].numel())
