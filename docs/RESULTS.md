# Benchmark results

Written by benchmarks/mnist_scaware_vs_float.py; numbers are measured, not
projected. Evaluation is the bit-accurate exact path (real Sobol bitstreams,
real gate logic) on the first 2000 MNIST test images, MLP 784-128-10, 3 epoch(s), seed fixed.

## MNIST: SC-aware training vs float-then-map

Float32 reference accuracy (subset): 0.9425 (baseline model), 0.8705 (SC-aware model, float forward).

| method | N | rng budget | exact-path accuracy |
|---|---|---|---|
| float-then-map | 256 | unbounded | 0.4655 |
| sc-aware | 256 | unbounded | 0.8355 |
| float-then-map | 1024 | unbounded | 0.9395 |
| sc-aware | 1024 | unbounded | 0.8690 |
| float-then-map | 256 | 2 | 0.0875 |
| sc-aware | 256 | 2 | 0.0875 |
| float-then-map | 1024 | 2 | 0.0875 |
| sc-aware | 1024 | 2 | 0.0875 |

## Verdict

SC-aware training beats float-then-map at the short stream length (N=256) with independent generators, which is the regime the thesis is about; at N=1024 the float-then-map model's higher float ceiling wins, the expected trade. At the 2-generator budget both methods collapse to chance: sharing one randomness source across a whole layer's activation and weight streams replaces the inner product with a distance-like function that neither training method in this benchmark survives (see docs/design_notes.md); that condition is reported as the honest limit of the correlation penalty, not as a win.

(Total benchmark wall time: 145 s. The rng budget of 2 forces the
second layer's activation and weight streams onto the same physical
generator: the correlated-multiply regime the correlation penalty exists
for. The unbounded budget gives every port its own generator. Both models
receive identical post-training per-layer output-gain calibration (the
standard SC dynamic-range scaling practice), so neither side wins by the
other being starved of dynamic range. Accumulation is APC, as in published
SC accelerators; single seed, single run, CPU.)
