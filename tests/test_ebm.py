"""EBM tests: Boltzmann stationarity, CD learning, reproducibility, energy.

The sampler's stationary distribution must match the analytic Boltzmann
distribution on a small system, contrastive divergence must reduce data
energy and recover planted structure, and everything must be reproducible
under a fixed seed. These are the Phase-2 conscience tests.
"""

from itertools import product

import torch

from scgrad.ebm import IsingLayer, contrastive_divergence, gibbs_sample
from scgrad.encoding import SCConfig

_REF_J = torch.tensor(
    [[0.0, 0.6, -0.3, 0.1], [0.6, 0.0, 0.4, -0.2], [-0.3, 0.4, 0.0, 0.5], [0.1, -0.2, 0.5, 0.0]]
)
_REF_H = torch.tensor([0.2, -0.1, 0.3, 0.0])


def _ref_layer() -> IsingLayer:
    cfg = SCConfig(seed=3, source="sobol")
    layer = IsingLayer(4, config=cfg)
    with torch.no_grad():
        layer.raw_coupling.copy_(_REF_J)
        layer.field.copy_(_REF_H)
    return layer


def _state_index(states: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([8.0, 4.0, 2.0, 1.0])
    return (((states + 1) / 2) * weights).sum(-1).long()


def test_coupling_symmetric_zero_diagonal() -> None:
    layer = _ref_layer()
    j = layer.coupling()
    torch.testing.assert_close(j, j.t())
    assert torch.all(torch.diag(j) == 0.0)


def test_energy_hand_computed() -> None:
    """Energy of a known 3-spin configuration matches the closed form."""
    cfg = SCConfig(seed=1)
    layer = IsingLayer(3, config=cfg)
    with torch.no_grad():
        layer.raw_coupling.copy_(torch.tensor([[0.0, 0.5, 0.0], [0.5, 0.0, 0.2], [0.0, 0.2, 0.0]]))
        layer.field.copy_(torch.tensor([0.1, 0.0, -0.3]))
    s = torch.tensor([1.0, -1.0, 1.0])
    j = layer.coupling()
    expected = -0.5 * (s @ j @ s) - layer.field @ s
    torch.testing.assert_close(layer.energy(s), expected)


def test_energy_batch_matches_single() -> None:
    layer = _ref_layer()
    states = torch.tensor(list(product([-1.0, 1.0], repeat=4)))
    batched = layer.energy(states)
    single = torch.stack([layer.energy(s) for s in states])
    torch.testing.assert_close(batched, single)


def test_stationary_distribution_matches_boltzmann() -> None:
    """Empirical Gibbs distribution is within TV 0.08 of analytic Boltzmann."""
    layer = _ref_layer()
    states = torch.tensor(list(product([-1.0, 1.0], repeat=4)))
    analytic = torch.softmax(-layer.energy(states), dim=0)
    samples = gibbs_sample(layer, n_steps=3000, n_chains=16, burn_in=200)
    flat = samples.value.reshape(-1, 4)
    counts = torch.bincount(_state_index(flat), minlength=16).float()
    empirical = counts / counts.sum()
    tv = 0.5 * (empirical[_state_index(states)] - analytic).abs().sum().item()
    assert tv < 0.08, f"TV distance {tv} exceeds 0.08"


def test_samples_are_bipolar_and_shaped() -> None:
    layer = _ref_layer()
    out = gibbs_sample(layer, n_steps=10, n_chains=5, burn_in=3, keep_every=2)
    vals = out.value
    assert vals.shape == (5, 5, 4)
    assert torch.all((vals == 1.0) | (vals == -1.0))


def test_reproducible_under_seed() -> None:
    layer = _ref_layer()
    a = gibbs_sample(layer, n_steps=50, n_chains=4)
    b = gibbs_sample(layer, n_steps=50, n_chains=4)
    assert torch.equal(a.value, b.value)


def test_different_seed_differs() -> None:
    layer1 = _ref_layer()
    layer2 = IsingLayer(4, config=SCConfig(seed=99, source="sobol"))
    with torch.no_grad():
        layer2.raw_coupling.copy_(_REF_J)
        layer2.field.copy_(_REF_H)
    a = gibbs_sample(layer1, n_steps=50, n_chains=4)
    b = gibbs_sample(layer2, n_steps=50, n_chains=4)
    assert not torch.equal(a.value, b.value)


def test_beta_zero_is_unbiased_coin() -> None:
    """At beta=0 spins flip independently of the field; means near zero."""
    layer = _ref_layer()
    out = gibbs_sample(layer, n_steps=2000, n_chains=16, beta=0.0)
    mean = out.value.mean().item()
    assert abs(mean) < 0.05, mean


def test_init_passthrough_stays_valid() -> None:
    layer = _ref_layer()
    init = torch.ones(8, 4)
    out = gibbs_sample(layer, n_steps=5, n_chains=8, init=init)
    assert torch.all((out.value == 1.0) | (out.value == -1.0))


def test_cd_reduces_energy_and_recovers_coupling() -> None:
    """CD-1 lowers mean data energy and learns the planted spin-pair coupling."""
    torch.manual_seed(0)
    cfg = SCConfig(seed=3, source="sobol")
    data = torch.where(torch.rand(64, 4) < 0.5, -1.0, 1.0)
    data[:, 1] = data[:, 0]
    model = IsingLayer(4, config=cfg)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    e0 = model.energy(data).mean().item()
    for step in range(60):
        loss = contrastive_divergence(model, data, k=1, id_offset=step * 300)
        opt.zero_grad()
        loss.backward()
        opt.step()
    e1 = model.energy(data).mean().item()
    assert e1 < e0 - 0.5, (e0, e1)
    j = model.coupling()
    assert j[0, 1].item() > 0.5
    off_diag = j.abs() - torch.diag(torch.diag(j.abs()))
    assert j[0, 1].abs() == off_diag.max(), "planted pair is the strongest coupling"


def test_cd_gradients_reach_parameters() -> None:
    cfg = SCConfig(seed=3, source="sobol")
    data = torch.where(torch.rand(16, 4) < 0.5, -1.0, 1.0)
    model = IsingLayer(4, config=cfg)
    loss = contrastive_divergence(model, data, k=1)
    assert loss.requires_grad
    loss.backward()
    assert model.raw_coupling.grad is not None
    assert model.field.grad is not None
    assert torch.any(model.raw_coupling.grad != 0.0)
