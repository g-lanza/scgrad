"""Dual-path tests: the approximate path against the bit-accurate exact path.

The conscience of the repo. For random valid inputs the approximate
(differentiable, eval-mode, noise off) forward of each layer family must
agree with the exact path (real bitstreams, real gate logic, counted
back) within the analytic tolerance from accuracy.py at every stream
length; the error must shrink on the 1/sqrt(N) trend; scales must match
bitwise in every case; and a shared-generator configuration must make
the exact path reveal the true correlated multiply output rather than
the intended product.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from scgrad.accuracy import tolerance
from scgrad.correlation import scc
from scgrad.encoding import SCConfig, SCNumber, encode
from scgrad.eval_exact import exact_forward
from scgrad.hardware import make_source
from scgrad.layers import SCConv2d, SCLinear, SCReLU

LENGTHS = [64, 256, 1024, 4096]
SEED = 7
WEIGHT_SEED = 1234
INPUT_SEED = 5678


def sobol_config(length: int, **overrides: object) -> SCConfig:
    """A deterministic bipolar Sobol config at the given stream length."""
    defaults: dict[str, object] = {
        "encoding": "bipolar",
        "length": length,
        "source": "sobol",
        "seed": SEED,
    }
    defaults.update(overrides)
    return SCConfig(**defaults)  # type: ignore[arg-type]


def rand_bipolar(shape: tuple[int, ...]) -> Tensor:
    """Deterministic uniform tensor strictly inside (-1, 1) so the clamp is identity."""
    gen = torch.Generator().manual_seed(INPUT_SEED)
    return torch.rand(shape, generator=gen) * 1.8 - 0.9


def dual_path_error(model: nn.Sequential, x: Tensor, config: SCConfig) -> float:
    """Mean |approx - exact| of the physical output values; asserts scale parity."""
    model.eval()
    with torch.no_grad():
        approx = model(x)
    assert isinstance(approx, SCNumber)
    exact = exact_forward(model, x, config)
    assert approx.value.shape == exact.value.shape
    assert approx.scale == exact.scale
    return float((approx.value - exact.value).abs().mean().item())


def linear_model(config: SCConfig) -> nn.Sequential:
    """A bare SCLinear(8 -> 6, bias) with weights fixed by WEIGHT_SEED."""
    torch.manual_seed(WEIGHT_SEED)
    return nn.Sequential(SCLinear(8, 6, bias=True, config=config))


def mlp_model(config: SCConfig) -> nn.Sequential:
    """An 8 -> 6 -> 4 MLP with SCReLU, weights fixed by WEIGHT_SEED."""
    torch.manual_seed(WEIGHT_SEED)
    return nn.Sequential(
        SCLinear(8, 6, bias=True, config=config),
        SCReLU(),
        SCLinear(6, 4, bias=True, config=config),
    )


def conv_model(config: SCConfig) -> nn.Sequential:
    """A bare SCConv2d(1 -> 2, 3x3, bias) with weights fixed by WEIGHT_SEED."""
    torch.manual_seed(WEIGHT_SEED)
    return nn.Sequential(SCConv2d(1, 2, 3, bias=True, config=config))


def test_sclinear_exact_matches_approx_within_tolerance() -> None:
    """Single SCLinear: per-N error bound at depth 1 and the 1/sqrt(N) trend."""
    x = rand_bipolar((4, 8))
    errors: dict[int, float] = {}
    for n in LENGTHS:
        cfg = sobol_config(n)
        err = dual_path_error(linear_model(cfg), x, cfg)
        assert err <= tolerance(n, depth=1), f"N={n}: err={err} > tol={tolerance(n, depth=1)}"
        errors[n] = err
    assert errors[4096] <= 0.5 * errors[64], f"no 1/sqrt(N) trend: {errors}"


def test_mlp_exact_matches_approx_within_tolerance() -> None:
    """Two-layer MLP with SCReLU: per-N error bound at depth 2 and the trend."""
    x = rand_bipolar((4, 8))
    errors: dict[int, float] = {}
    for n in LENGTHS:
        cfg = sobol_config(n)
        err = dual_path_error(mlp_model(cfg), x, cfg)
        assert err <= tolerance(n, depth=2), f"N={n}: err={err} > tol={tolerance(n, depth=2)}"
        errors[n] = err
    assert errors[4096] <= 0.5 * errors[64], f"no 1/sqrt(N) trend: {errors}"


def test_scconv2d_exact_matches_approx_within_tolerance() -> None:
    """Single SCConv2d on a 6x6 input: per-N error bound at depth 1 and the trend."""
    x = rand_bipolar((2, 1, 6, 6))
    errors: dict[int, float] = {}
    for n in LENGTHS:
        cfg = sobol_config(n)
        err = dual_path_error(conv_model(cfg), x, cfg)
        assert err <= tolerance(n, depth=1), f"N={n}: err={err} > tol={tolerance(n, depth=1)}"
        errors[n] = err
    assert errors[4096] <= 0.5 * errors[64], f"no 1/sqrt(N) trend: {errors}"


def test_sclinear_lfsr_within_tolerance() -> None:
    """Single SCLinear on the LFSR source stays inside the depth-1 bound."""
    x = rand_bipolar((4, 8))
    for n in [1024, 4096]:
        cfg = sobol_config(n, source="lfsr")
        err = dual_path_error(linear_model(cfg), x, cfg)
        assert err <= tolerance(n, depth=1), f"N={n}: err={err} > tol={tolerance(n, depth=1)}"


def test_scale_parity_is_exact_for_stacked_layers() -> None:
    """Approx and exact scales are bitwise equal and match the fan-in product."""
    cfg = sobol_config(256)
    model = mlp_model(cfg)
    model.eval()
    x = rand_bipolar((2, 8))
    with torch.no_grad():
        approx = model(x)
    assert isinstance(approx, SCNumber)
    exact = exact_forward(model, x, cfg)
    assert approx.scale == exact.scale
    assert approx.scale == (1.0 / 9) / 7


def test_shared_generator_exact_multiply_reveals_correlation() -> None:
    """With n_rngs=1 the physical XNOR lands on the correlated closed form.

    Two bipolar scalars 0.5 and -0.5 share one generator: the comparator
    streams are maximally correlated (SCC near +1) and the gated multiply
    counts to 1 - 2|p_a - p_b| = 0.0 in value space, not the intended
    product -0.25. The exact path tells the truth about correlation.
    """
    cfg = sobol_config(4096, n_rngs=1)
    a = encode(torch.tensor(0.5), cfg)
    b = encode(torch.tensor(-0.5), cfg)
    source = make_source(cfg)
    a_bits = source.bits(a.probabilities(), cfg.length, a.corr_id)
    b_bits = source.bits(b.probabilities(), cfg.length, b.corr_id)
    assert float(scc(a_bits, b_bits).item()) > 0.9
    xnor = a_bits == b_bits
    v_hat = 2.0 * float(xnor.to(torch.float64).mean().item()) - 1.0
    correlated = 1.0 - 2.0 * abs(0.75 - 0.25)
    product = 0.5 * -0.5
    assert abs(v_hat - correlated) <= 0.05, f"v_hat={v_hat} far from correlated {correlated}"
    assert abs(v_hat - product) >= 0.15, f"v_hat={v_hat} suspiciously near product {product}"
