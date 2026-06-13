# scgrad

Differentiable stochastic-computing primitives for PyTorch: train neural
networks natively SC-aware, and check every claim against real bitstreams.

Stochastic computing (Gaines, 1967) represents a number as the probability
of a 1 in a random bitstream: multiplication is a single AND gate (XNOR for
signed values), addition is a multiplexer, and precision is a knob you turn
by counting more bits. The standard practice today trains a network in
float32, maps it to SC hardware, and accepts the loss. scgrad moves the SC
arithmetic inside training: forward passes compute what the SC circuit
produces in expectation, differentiably, with the counting noise of finite
stream length injected so the optimizer feels it, and a correlation penalty
for streams forced to share randomness.

To our knowledge this is the first pip-installable PyTorch library that
implements SC bitstream operations as differentiable autograd primitives
for natively SC-aware training, paired with a bit-accurate reference path
for honest evaluation. The closest prior art trains float-then-maps:
UnarySim (Wu et al., a forward cycle-accurate simulator) and Rosselló et
al. 2024 (arithmetic-aware training, MDPI Electronics 13:2846). The claim
is about the packaging — differentiable primitives, a differentiable
correlation penalty, and a dual exact/approximate architecture — not about
inventing SC-aware training as a concept.

## Install

```
git clone <this repo> && cd scgrad
uv sync                  # core
uv sync --extra gui      # + the terminal instrument
```

## Quickstart

```python
import torch
from torch import nn
from scgrad import SCConfig, SCLinear, SCReLU, evaluate_exact

config = SCConfig(encoding="bipolar", length=256, noise=True, accumulator="apc")
model = nn.Sequential(
    SCLinear(784, 128, config=config),
    SCReLU(),
    SCLinear(128, 10, config=config),
)
out = model(torch.rand(32, 784) * 2 - 1)   # SCNumber: value, scale, stream id
loss = nn.functional.cross_entropy(out.value / out.scale, torch.randint(0, 10, (32,)))
loss.backward()                             # ordinary autograd, SC-shaped loss surface
# after training: score the real circuit, real bitstreams, real gates
# accuracy = evaluate_exact(model, test_loader, config)["accuracy"]
```

## What is in the box

- `encoding` — `SCNumber`/`SCConfig`: values in probability space with
  scale-factor and stream-identity metadata. Bipolar multiply is XNOR;
  MUX addition is a scaled average and the scale is tracked, not hidden.
- `ops` — `sc_mul`, `sc_add`, `sc_add_tree` as `torch.autograd.Function`s,
  each gated by float64 gradcheck.
- `layers` — `SCLinear`, `SCConv2d`: drop-in layers with MUX or APC
  accumulation, analytic counting-noise injection, and per-port stream
  identities fixed at construction, the way SNGs are silicon.
- `hardware` + `eval_exact` — the conscience of the repo: LFSR and Sobol
  bitstream generators and a bit-accurate forward that replays the circuit
  on real streams. `tests/test_dual_path.py` proves the two paths converge
  as stream length grows; the tolerance comes from the analytic error
  model, never from tuning.
- `correlation` — the Alaghi–Hayes SCC measured on real bits, plus a
  differentiable penalty equal to the closed-form error of a shared-RNS
  multiply, validated by rank-correlation against exact-path measurements.
- `accuracy` — length-to-error estimator (analytic plus empirical
  calibration) used by users to pick N and by the test suite as tolerance.
- `ebm` — phase 2: an Ising layer, Gibbs sampling driven by the same
  bitstream generators (the p-bit picture), and contrastive-divergence
  training. A PyTorch-native, bitstream-grounded counterpart to the JAX
  tools in this space (THRML, thermox); a software prototyping substrate,
  not a hardware claim.
- `gui` — `scgrad-gui`: a dense terminal instrument over a live SC training
  run (live bitstream raster, SCC matrix, error curve, loss decomposition).

## Measured results

From `benchmarks/mnist_scaware_vs_float.py` (bit-accurate evaluation, 2000
MNIST test images, identical 784-128-10 circuit, identical gain calibration
for both sides; full table in `docs/RESULTS.md`):

| | N=256 | N=1024 |
|---|---|---|
| float-then-map | 0.466 | 0.940 |
| SC-aware (this library) | 0.836 | 0.869 |

SC-aware training wins decisively at short stream length — the regime SC
exists for — and pays for its noise robustness with a lower float ceiling
at long lengths. With a pathological randomness budget (one generator
shared across a whole layer) both methods collapse; the limit is reported,
not hidden. Single seed, single run, CPU; treat the numbers as a
demonstration, not a survey.

## Docs

`docs/theory.md` derives the math with sources; `docs/design_notes.md` is
the engineering log, including the bugs found and the honest limitations
(partial-correlation propagation is not modeled; activations are computed
in the decoded domain in v0.1).

## License

MIT.
