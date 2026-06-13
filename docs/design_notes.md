# Design notes — the engineering log

Decisions made while building, and why. Newest entries last. This is the
project's memory; read it before changing anything load-bearing.

## Scale composition rule

MUX addition needs a scale-factor bookkeeping rule that composes. We use:
for a MUX with select weights w_i over inputs with scales s_i, output
scale = mean_i(w_i * s_i). When every per-term factor agrees (the
balanced case: equal scales, uniform selection) decode(descale=True)
recovers the exact intended sum; mismatched factors give a weighted
approximation, stated in the sc_add docstring. This single rule covers
the 2-input MUX, the k-way tree, and the layer fan-in case, and it makes
the layer's pre-scaled-bias trick (below) exact.

## Bias pre-scaling, and clamp order

A layer's MUX terms are products x_j * w_j carrying the input scale s_in;
the bias term would carry scale 1 and break recovery. We program the bias
register in the incoming scale (b_enc = clamp(s_in * b)) so every term
carries the same factor and decode stays exact. Clamp comes after
scaling: the register holds an encodable value. The exact path does the
same arithmetic; the dual-path test fails if either side reorders these.

## The Sobol joint-uniformity bug

First implementation gave each correlation id its own scrambled 1-D
Sobol engine. Marginals were perfect; XNOR multiplication was wrong
(measured v near 0 where the product should be -0.24). Two separately
constructed low-discrepancy sequences are not jointly uniform on the
square, and multiplication correctness is a property of the pair, not
the marginals. Fix: one Sobol sequence, one dimension per generator
index — joint equidistribution across dimensions is what Sobol is for.
The lesson is recorded in docs/theory.md section 6. The LFSR source did
not have this failure (phase-shifted m-sequences have near-ideal pairwise
lag properties), only a slower convergence rate.

Residual limitation, documented rather than fixed: unscrambled Sobol at
exactly p = 1/2 yields the alternating stream regardless of the
fast-forward seed, so seed variation does not produce replicate streams
at dyadic probabilities. That is the deterministic-SC school's feature
(zero-variance estimates), not a defect; corr_ids, not seeds, are the
independence mechanism.

## Correlation identities are silicon, not bookkeeping

Ports (input SNG, weight SNG, bias SNG, output regenerator) get
persistent corr_ids at layer construction, because SNGs are fixed
hardware. An early version minted a fresh id per forward inside sc_relu,
which made the rng-budget collision structure nondeterministic across
steps; the penalty silently averaged over different circuits. ReLU is a
digital-domain operation on the counted value, upstream of the output
SNG, so it carries scale AND corr_id through unchanged; both paths agree.

## The correlation penalty's form

The differentiable proxy is the closed-form error of the SCC = +1
collision: |min(p_a,p_b) - p_a p_b| (unipolar AND) and
|(1 - 2|p_a - p_b|) - v_a v_b| (bipolar XNOR), summed over multiplies
whose streams share a physical generator under SCConfig.n_rngs. This is
the actual hardware error of a shared-RNS multiply, differentiable a.e.
in the parameters, zero when nothing collides, and it rank-correlates
near 1.0 with exact-path measurements (tests/test_correlation.py). v0.1
deliberately does not model partial correlation propagating through op
outputs (layer outputs are regenerated streams, the standard
decorrelation practice); that is the known limitation of the proxy.

## Training noise is the mechanism, not a flourish

A pure expectation forward carries no SC imprecision, so an optimizer
feels nothing of finite N. With config.noise, layers inject the analytic
counting noise of their output stream (reparameterized, detached std).
This is the load-bearing piece of SC-aware training; the MNIST benchmark
turns it on for the SC-aware contender and the effect is visible in the
results table.

## APC accumulation

Pure MUX accumulation at fan-in 785 attenuates the signal by 1/785 while
counting noise stays ~1/sqrt(N): at N = 256 the benchmark was chance for
both contenders — a degenerate comparison, not a thesis test. Published
SC accelerators (SC-DCNN, uGEMM) use accumulative parallel counters for
wide sums. SCConfig.accumulator = "apc" models that: same expected value,
same scale bookkeeping, variance smaller by the fan-in factor (counts
every product bit every clock). MUX remains the default and the
Gaines-pure primitive; sc_add/sc_add_tree are MUX-only by design — APC
is a layer-level accumulator.

## Output gain registers

First quick benchmark run: SC-aware 0.59 vs float-then-map 0.085 at
N = 256. Honest inspection showed the baseline was dying of dynamic-range
loss (1/k attenuation pushing values under the noise floor), not of
SC noise per se — a strawman. Real SC designs apply per-layer binary-
domain scaling after the counter. layers now carry output_gain (a
fixed-point multiply with saturation, mirrored exactly in the exact
path), and the benchmark calibrates gains identically for both
contenders post-training from real activation quantiles. Whatever margin
survives that is real.

## Exact-path vectorization

Two tricks keep the bit-accurate path tractable without per-element
Python loops. (1) Sources expose uniforms(n, id); a MUX stage gathers
only the selected term's bits via the comparator inequality instead of
materializing (B, K, N) streams. (2) Counting coinciding ones across
time (and, for APC, across terms) is a matmul over the flattened
(term, time) axis; counts stay below 2^24 and are exact in float32.
Blocked over time chunks and batch rows to bound memory.

## What the benchmark actually showed

Three-epoch run, 2000 test images, identical circuits, identical gain
calibration (docs/RESULTS.md has the table). At N = 256 with independent
generators, SC-aware training reaches 0.836 against the float-then-map
0.466: the noise injection is doing the work, and the SC-aware model
nearly attains its own float ceiling (0.871) at a quarter of the stream
length the baseline needs. At N = 1024 the baseline's higher float
ceiling (0.943) wins — SC-aware training trades peak float accuracy for
robustness, which is the correct trade exactly when N is short, and
short N is the point of SC. At a 2-generator budget both collapse to
chance: sharing one RNS across a layer's activation and weight tensors
replaces every product with the correlated closed form (a distance, not
a product), and no penalty recovers an inner product that the hardware
no longer computes. The honest fix is a correlation-aware forward that
trains through the correlated closed form instead of penalizing it —
noted as future work, not attempted in v0.1.

## EBM substrate decisions (Phase 2)

Chains get one generator id per (chain, spin) site: shared randomness
plus identical kernels coalesce chains, so ensemble statistics need
distinct streams; a finite rng budget correlating chains is then a
modeling feature, not a bug. The Sobol fast path draws all site columns
from one engine in a single pass — a Sobol dimension's values do not
depend on the engine's total dimension, so this is exactly equal to
per-site uniforms() calls. CD training passes id_offset per step so the
negative phase does not reuse one fixed noise sequence every step.
Verified against ground truth: 4-spin empirical distribution within
TV = 0.036 of the analytic Boltzmann distribution; CD-1 recovers a
planted spin-pair coupling.

## Toolchain notes

mypy strict with torch needs a handful of targeted
`# type: ignore[no-untyped-call]` comments (Function.apply, SobolEngine,
fast_forward, Tensor.backward) — these are torch's typing gaps, never
ours; no module-level ignores for first-party code.

## Sign-off status

Both phases are green end to end. The full suite is 143 tests across
encoding, hardware, ops (float64 gradcheck gate), accuracy, correlation,
layers, the dual-path invariant, property-based checks, APC, the
conv/flatten/linear pipeline, the EBM, and the GUI smoke test; ruff and
mypy strict are clean across the tree. The package builds a wheel that
installs and imports in a fresh venv. All three examples run start to
finish, the GUI renders headless with live data, and the MNIST thesis
benchmark populates RESULTS.md with the measured verdict. The git history
is free of any AI attribution. Tagged v0.1.0 (Phase 1) and v0.2.0
(Phase 2). Remaining honest gaps, all documented above: partial-
correlation propagation through op outputs is not modeled; activations
are computed in the decoded domain; the shared-RNS collapse regime is
reported, not solved (a correlation-aware forward is the future-work
fix).
