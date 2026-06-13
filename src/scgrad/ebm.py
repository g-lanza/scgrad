"""Energy-based models on the SC substrate: Ising layer, Gibbs, CD training.

The bridge from classical bitstream SC to energy-based sampling. Spins
are bipolar SC values (+1/-1); every stochastic update bit is generated
by the same hardware.py bitstream sources the rest of the library uses
(the p-bit picture: a spin is a stochastic bit whose probability is set
by its local field, and the randomness is a physical generator subject
to the same seeding, reproducibility, and rng-budget semantics as every
other stream). The local-field arithmetic in this version is digital
float, as in p-bit controllers; pushing it into SC gates is future work.

Positioning, honestly: this is a software prototyping substrate in the
spirit of THRML and thermox (which are JAX, and operate on spins and
OU processes); scgrad's version is PyTorch-native and grounded in
Gaines-lineage bitstream generation. It makes no hardware claim.

Each chain uses its own generator ids (chains sharing a generator would
coalesce: identical randomness plus identical kernels merge chains).
Generator indices live in the same budget space as the rest of the
library, so a finite SCConfig.n_rngs realistically correlates chains.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from scgrad.encoding import SCConfig, SCNumber, fresh_corr_id
from scgrad.hardware import make_source


class IsingLayer(nn.Module):
    """An Ising/Boltzmann energy over bipolar spins; couplings as parameters.

    E(s) = -1/2 s^T J s - h^T s with J symmetric and zero-diagonal (both
    enforced structurally from the raw parameter, so gradients respect
    the constraint instead of fighting it).
    """

    def __init__(self, n_spins: int, config: SCConfig | None = None) -> None:
        super().__init__()
        self.n_spins = n_spins
        self.config = config if config is not None else SCConfig()
        self.raw_coupling = nn.Parameter(torch.randn(n_spins, n_spins) * 0.1)
        self.field = nn.Parameter(torch.zeros(n_spins))
        self.base_corr_id = fresh_corr_id()

    def coupling(self) -> Tensor:
        """Symmetric, zero-diagonal coupling matrix."""
        j = (self.raw_coupling + self.raw_coupling.t()) / 2.0
        return j - torch.diag(torch.diag(j))

    def energy(self, spins: Tensor) -> Tensor:
        """Energy of each configuration; spins is (..., n_spins) in {-1,+1}."""
        j = self.coupling()
        quad = torch.einsum("...i,ij,...j->...", spins, j, spins)
        lin = spins @ self.field
        return -0.5 * quad - lin


def _chain_uniforms(
    layer: IsingLayer, config: SCConfig, n_chains: int, n_sweeps: int, offset: int
) -> Tensor:
    """(n_chains, n_spins, n_sweeps) update uniforms from the bitstream sources.

    Chain c, spin i draws from generator id base + offset + c * n_spins + i:
    one physical generator per (chain, spin) site, deterministic under the
    config seed, independent across sites up to the configured rng budget.
    For the Sobol source all site columns are drawn from one engine in a
    single pass (a Sobol dimension's values do not depend on the engine's
    total dimension, so this equals per-site uniforms() calls exactly).
    """
    from scgrad.hardware import _SOBOL_MAX_DIMS, SobolSource

    source = make_source(config)
    ids = [
        layer.base_corr_id + offset + c * layer.n_spins + i
        for c in range(n_chains)
        for i in range(layer.n_spins)
    ]
    if isinstance(source, SobolSource):
        cols = [config.rng_index(i) % _SOBOL_MAX_DIMS for i in ids]
        engine = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
            dimension=max(cols) + 1, scramble=False
        )
        engine.fast_forward(1 + source.seed * 64)  # type: ignore[no-untyped-call]
        draws = engine.draw(n_sweeps, dtype=torch.float64)
        stacked = draws[:, cols].t()
    else:
        stacked = torch.stack([source.uniforms(n_sweeps, i) for i in ids])
    return stacked.reshape(n_chains, layer.n_spins, n_sweeps)


def gibbs_sample(
    layer: IsingLayer,
    n_steps: int,
    config: SCConfig | None = None,
    n_chains: int = 32,
    beta: float = 1.0,
    init: Tensor | None = None,
    burn_in: int = 0,
    keep_every: int = 1,
    id_offset: int = 0,
) -> SCNumber:
    """Systematic-scan Gibbs sampling with SC bitstream randomness.

    Each sweep updates every spin once, vectorized across chains:
    p(s_i = +1 | rest) = sigmoid(2 beta (J_i . s + h_i)), and the update
    bit is the comparator inequality (r < p) against the site's bitstream
    generator, i.e. an SNG bit whose probability is the conditional.
    Returns the kept states stacked as a bipolar SCNumber of shape
    (n_kept, n_chains, n_spins). Deterministic given config.seed.
    """
    cfg = config if config is not None else layer.config
    j = layer.coupling().detach()
    h = layer.field.detach()
    if init is None:
        gen = torch.Generator()
        gen.manual_seed((cfg.seed if cfg.seed is not None else 0) * 7919 + 17)
        spins = torch.where(torch.rand(n_chains, layer.n_spins, generator=gen) < 0.5, -1.0, 1.0)
    else:
        spins = init.clone().float()
    total = burn_in + n_steps
    uniforms = _chain_uniforms(layer, cfg, n_chains, total, id_offset)
    kept = []
    for t in range(total):
        for i in range(layer.n_spins):
            local = spins @ j[i] + h[i]
            p_up = torch.sigmoid(2.0 * beta * local)
            spins[:, i] = torch.where(uniforms[:, i, t] < p_up, 1.0, -1.0)
        if t >= burn_in and (t - burn_in) % keep_every == 0:
            kept.append(spins.clone())
    states = torch.stack(kept)
    return SCNumber(states, cfg, scale=1.0, corr_id=layer.base_corr_id)


def contrastive_divergence(
    layer: IsingLayer,
    data: Tensor,
    k: int = 1,
    beta: float = 1.0,
    config: SCConfig | None = None,
    id_offset: int = 0,
) -> Tensor:
    """CD-k loss: E[data] - E[model samples], negative phase from the SC sampler.

    Hinton 2002: approximate the likelihood gradient by contrasting the
    data with k Gibbs steps started at the data. The returned scalar's
    gradient with respect to the layer parameters is the CD-k gradient
    (sample states are detached constants; only the energies carry
    gradients).
    """
    cfg = config if config is not None else layer.config
    negative = gibbs_sample(
        layer,
        n_steps=1,
        config=cfg,
        n_chains=data.shape[0],
        beta=beta,
        init=data,
        burn_in=k - 1,
        id_offset=id_offset,
    ).value[-1]
    return layer.energy(data).mean() - layer.energy(negative.detach()).mean()
