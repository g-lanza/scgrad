"""Sample an Ising EBM with SC bitstreams, then learn one back with CD.

Two demonstrations on the same substrate:

1. Sampling. Build a known 4-spin Ising model, draw samples with SC
   bitstream-driven Gibbs sampling, and show the empirical distribution
   converges to the analytic Boltzmann distribution. The randomness is
   the same hardware.py generator the rest of the library uses.

2. Learning. Generate data from a ground-truth model, train a fresh
   IsingLayer with contrastive divergence using the SC sampler for the
   negative phase, and show the recovered couplings match the truth.

Run: uv run python examples/03_ebm_ising_sampling.py
"""

from itertools import product

import torch

from scgrad.ebm import IsingLayer, contrastive_divergence, gibbs_sample
from scgrad.encoding import SCConfig

torch.manual_seed(0)
config = SCConfig(seed=3, source="sobol")

truth = IsingLayer(4, config=config)
with torch.no_grad():
    truth.raw_coupling.copy_(
        torch.tensor(
            [
                [0.0, 0.6, -0.3, 0.1],
                [0.6, 0.0, 0.4, -0.2],
                [-0.3, 0.4, 0.0, 0.5],
                [0.1, -0.2, 0.5, 0.0],
            ]
        )
    )
    truth.field.copy_(torch.tensor([0.2, -0.1, 0.3, 0.0]))

print("1. SC Gibbs sampling vs analytic Boltzmann distribution")
states = torch.tensor(list(product([-1.0, 1.0], repeat=4)))
analytic = torch.softmax(-truth.energy(states), dim=0)

samples = gibbs_sample(truth, n_steps=4000, n_chains=16, burn_in=200)
flat = samples.value.reshape(-1, 4)
weights = torch.tensor([8.0, 4.0, 2.0, 1.0])
idx = (((flat + 1) / 2) * weights).sum(-1).long()
counts = torch.bincount(idx, minlength=16).float()
empirical = counts / counts.sum()
state_idx = (((states + 1) / 2) * weights).sum(-1).long()
tv = 0.5 * (empirical[state_idx] - analytic).abs().sum().item()

print(f"  {'state':>10}  {'analytic':>9}  {'empirical':>9}")
order = torch.argsort(analytic, descending=True)[:6]
for k in order:
    spins = "".join("+" if v > 0 else "-" for v in states[k])
    print(f"  {spins:>10}  {analytic[k].item():9.4f}  {empirical[state_idx[k]].item():9.4f}")
print(f"  total-variation distance: {tv:.4f}\n")

print("2. Contrastive-divergence learning from samples")
data = gibbs_sample(truth, n_steps=400, n_chains=64, burn_in=100).value.reshape(-1, 4)
model = IsingLayer(4, config=config)
opt = torch.optim.Adam(model.parameters(), lr=5e-2)
for step in range(200):
    batch = data[torch.randint(0, data.shape[0], (64,))]
    loss = contrastive_divergence(model, batch, k=1, id_offset=step * 300)
    opt.zero_grad()
    loss.backward()
    opt.step()

learned = model.coupling()
true_j = truth.coupling()
print(f"  {'pair':>6}  {'true J':>8}  {'learned J':>9}")
for i, j in ((0, 1), (1, 2), (2, 3), (0, 2)):
    print(f"  ({i},{j})  {true_j[i, j].item():8.3f}  {learned[i, j].item():9.3f}")
err = (learned - true_j).abs().mean().item()
print(f"  mean absolute coupling error: {err:.3f}")
