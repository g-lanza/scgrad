# Theory

The mathematics behind scgrad. Every formula here is implemented somewhere in
`src/scgrad/`, and the implementation is the contract: where the code makes a
choice the literature leaves open, the docstring says so and this document
repeats it. Citations refer to the list at the end.

## 1. The representation

A stochastic-computing number is a probability carried by a bitstream. A value
is encoded as a stream of $N$ bits, each independently 1 with a fixed
probability, and read back by counting (Gaines 1967).

Two encodings (`encoding.py`):

- **Unipolar**: $x \in [0, 1]$, $p = x$.
- **Bipolar**: $x \in [-1, 1]$, $p = (x + 1)/2$, so $x = 2p - 1$.

Counting $N$ Bernoulli($p$) bits gives the estimate $\hat{p} = n_1/N$ with

$$\mathrm{Var}[\hat{p}] = \frac{p(1-p)}{N} \le \frac{1}{4N},$$

so the standard error falls as $1/\sqrt{N}$: worst case $0.5/\sqrt{N}$ in
probability space, and twice that, $1/\sqrt{N}$, in bipolar value space because
the value map stretches the range by 2 (`accuracy.counting_std`). Precision is
bought with stream length, linearly in time and as the square root in accuracy.
This trade is the entire representational idea (Gaines 1967).

## 2. Multiplication

**Unipolar: AND.** For independent streams, $P(a_t \wedge b_t) = p_a p_b$. One
gate per bit multiplies.

**Bipolar: XNOR.** The XNOR of independent streams is 1 when both bits agree:

$$p^* = p_a p_b + (1 - p_a)(1 - p_b)$$

(Gaines 1968, the "Foundations" note). Mapping back to value space:

$$v^* = 2p^* - 1 = 2p_a p_b + 2(1-p_a)(1-p_b) - 1 = (2p_a - 1)(2p_b - 1) = v_a v_b.$$

So both encodings multiply in value space, which is why `ops.sc_mul` is a
single product $v_a v_b$ — it is the closed-form expectation of the gate
circuit on independent streams.

**Independence is the precondition.** Suppose the two streams come from
comparator SNGs sharing one random sequence $r_t$: $a_t = (r_t < p_a)$,
$b_t = (r_t < p_b)$. Then both bits fire exactly when $r_t$ falls below both
thresholds:

$$P(a_t \wedge b_t) = P(r_t < \min(p_a, p_b)) = \min(p_a, p_b),$$

not $p_a p_b$. For XNOR, the agreement probability is
$\min(p_a,p_b) + (1 - \max(p_a,p_b)) = 1 - |p_a - p_b|$, so the value computed
is

$$v^* = 2(1 - |p_a - p_b|) - 1 = 1 - 2\,|p_a - p_b|,$$

not $v_a v_b$. This is the maximal-correlation (SCC $= +1$) failure mode, and
it is exactly what a shared RNG produces on hardware. scgrad records every
multiply with the correlation tracker and shows the degraded result truthfully
on the exact path (sections 5 and 8).

## 3. Addition

Direct addition is impossible: $p_a + p_b$ can exceed 1, and no bitstream has a
bit probability above 1. SC addition is therefore a scaled average (Gaines
1969).

**Two-input MUX.** A select stream with $P(\text{select } a) = s$ passes $a$'s
bit with probability $s$, giving $p_{out} = s\,p_a + (1-s)\,p_b$. Because the
value map is affine, the same combination holds in value space:

$$v = s\,v_a + (1 - s)\,v_b.$$

**$k$-way uniform MUX.** A uniform selector over $k$ inputs gives

$$v = \frac{1}{k} \sum_{i=1}^{k} v_i,$$

the physical wire value. The intended sum is $k$ times larger.

**Scale bookkeeping.** scgrad keeps the physical value in `SCNumber.value` and
the accumulated factor in `SCNumber.scale`. The composition rule (`ops.py`):
for a MUX with select weights $w_i$ over inputs carrying scales $s_i$,

$$\text{scale}_{out} = \operatorname{mean}_i(w_i\, s_i).$$

When every per-term factor $w_i s_i$ equals a common $c$ (the balanced case),
the physical output is $c \sum_i (\text{intended}_i)$ and the scale is $c$, so
`decode(descale=True)` recovers the intended sum exactly. With mismatched
factors the recovery is the correspondingly weighted combination. For the
uniform $k$-way tree with common input scale $s$ this gives scale $s/k$, hence
the layers' output scale $s_{in}/k$ at fan-in $k$ (`layers.py`).

**APC alternative.** The accumulative parallel counter counts all $k$ product
bits every clock into a binary count, instead of selecting one term per clock.
Expected value and scale bookkeeping are identical to the MUX; the noise is
not. The MUX output is one counted stream, so its variance is the single-stream
counting variance $p(1-p)/N$ regardless of fan-in. The APC output is the
average of $k$ exactly counted streams:

$$\mathrm{Var}_{APC} = \frac{1}{k^2} \sum_{j=1}^{k} \frac{1 - v_j^2}{N}
\quad\text{(bipolar; } p_j(1-p_j)/N \text{ unipolar)},$$

smaller than the MUX by roughly the factor $k$ — worst-case std
$\mathrm{counting\_std}/\sqrt{k}$ (`accuracy.apc_counting_std`) — at the
hardware cost of a binary adder tree and of leaving the unary domain at the
accumulation point. APC accumulation is the practice of published SC
accelerators (Ren et al. 2017; Wu et al. 2020). `SCConfig.accumulator` selects
between the two; exposing this noise trade is part of why the library exists.

## 4. Error model

The analytic model in `accuracy.py`:

- **Counting noise.** Worst-case std $0.5/\sqrt{L}$ in probability space,
  $1/\sqrt{L}$ for bipolar values (section 1).
- **Scaled addition.** The uGEMM worst-case bound for a $k$-term scaled MUX
  add at stream length $L$ (Wu et al. 2020):

  $$\epsilon_{max} = \frac{k - 1}{L \cdot k} \ \text{(unipolar)}, \qquad
  \frac{2(k-1)}{L \cdot k} \ \text{(bipolar)}.$$

- **Depth composition.** Each layer re-counts a stream; independent zero-mean
  stage errors add in variance, so a circuit of depth $d$ scales the counting
  std by $\sqrt{d}$.

The dual-path test tolerance (`accuracy.tolerance`) is

$$\tau(N, d) = 5\sqrt{d}\cdot \mathrm{counting\_std}(N) + d \cdot 2^{-16},$$

five standard deviations of the composed counting noise plus a per-stage
comparator quantization floor (a 16-bit comparator quantizes $p$ to $2^{-16}$).
The mean absolute error of zero-mean noise sits near $0.8\sigma$, so the factor
of five is slack — but a systematic bug such as a forgotten scale factor is an
$O(1)$ discrepancy and still fails loudly. The tolerance is derived from this
model, never tuned to make a failing comparison pass.

## 5. Correlation

The standard correlation metric for bitstreams is the stochastic
cross-correlation SCC (Alaghi & Hayes 2013; Alaghi et al. 2019). With marginal
one-probabilities $p_1, p_2$ and joint $p_{11} = P(a_t = b_t = 1)$, and
$\delta = p_{11} - p_1 p_2$:

$$\mathrm{SCC} =
\begin{cases}
\dfrac{\delta}{\min(p_1, p_2) - p_1 p_2} & \delta > 0,\\[2ex]
\dfrac{\delta}{p_1 p_2 - \max(p_1 + p_2 - 1,\, 0)} & \delta \le 0,
\end{cases}$$

normalized so $+1$ is maximal positive correlation, $0$ independence, $-1$
maximal negative correlation. Degenerate streams (all zeros or all ones) have
no correlation freedom; their SCC is defined as 0. `correlation.scc` computes
this on real bitstreams for the exact path.

Sampled bits are not differentiable, so the training penalty is not the SCC
itself. Instead scgrad uses the closed-form error of the SCC $= +1$ collision —
the case a shared comparator RNS actually produces (section 2):

$$e_{uni} = \left|\min(p_a, p_b) - p_a p_b\right|, \qquad
e_{bi} = \left|\bigl(1 - 2|p_a - p_b|\bigr) - v_a v_b\right|.$$

This is exactly the hardware error the collision would cause at that multiply,
it is a composition of products, minima, and absolute values of the model
parameters — differentiable almost everywhere — and it is zero when no streams
collide under the configured randomness budget (`SCConfig.n_rngs`, which maps
correlation ids onto physical generators by `corr_id % n_rngs`).
`correlation_loss` sums the mean per-element error over colliding multiplies.
The proxy is validated by rank correlation against measured exact-path error in
`tests/test_correlation.py`. Known limitation, stated in the module docstring:
it models the maximal-correlation collision only; partial correlation arriving
through upstream op outputs is not modeled in v0.1.

## 6. Bitstream generation

All exact-path randomness flows through `hardware.py`. The generator is the
comparator SNG: a random-number sequence $r_t \in [0, 1)$ and the bit rule
$b_t = (r_t < p)$, which gives $E[b_t] = p$ for uniform $r_t$. One sequence
drives many comparators (one RNS, many SNGs), the standard hardware layout;
independence across tensors is governed by the correlation id.

**LFSR.** A maximal-length 16-bit Fibonacci LFSR, taps $(16, 15, 13, 4)$,
period $2^{16} - 1$. Distinct correlation ids are distinct phases of the one
m-sequence; m-sequence cross-correlation at nonzero lag is $-1/\text{period}$,
effectively zero (Hsiao et al. 2019). Beyond the period the sequence repeats,
exactly as the hardware register would.

**Sobol.** Low-discrepancy sequences converge much faster than pseudo-random
ones at equal length — this is the deterministic/low-discrepancy school of SC
(Riedel 2019; Najafi's line of work; Hsiao et al. 2019). The pitfall: marginal
uniformity is not enough. Multiplication correctness requires the pair
$(r_a(t), r_b(t))$ to equidistribute over the unit square, because
$P(a_t \wedge b_t) = P(r_a < p_a,\, r_b < p_b)$ equals $p_a p_b$ only if the
pair fills the square. Two separately constructed one-dimensional
low-discrepancy sequences do not do this — in the worst case they are the same
sequence, $r_a = r_b$, and the AND computes $\min(p_a, p_b)$ (section 2).
Sobol dimensions are designed for joint equidistribution, so scgrad maps each
generator index to its own dimension of one Sobol sequence, never to an
independently constructed 1-D sequence. The dimension budget is finite (1024),
a documented hardware-style limit.

## 7. Gradients

The approximate path needs no gradient tricks for the arithmetic itself: every
forward in `ops.py` and `layers.py` is the closed-form expectation of the gate
circuit on independent streams — $v_a v_b$ for the multiply, the affine
combination for the MUX, matmul-over-$k$ for the layers — and expectations of
these circuits are smooth in the values. Backwards are hand-written
(product rule, linear maps) and gated by float64 `torch.autograd.gradcheck`.

The one non-smooth point is the encode clamp. A hard clamp has zero gradient
outside the encoding range, which permanently stalls any weight that drifts
past it. scgrad uses the straight-through estimator there (Bengio et al. 2013):
forward is the clamp, backward passes the gradient unchanged. Inside the range
the forward is the identity, so gradcheck holds there; on the clamped region
the pass-through is deliberate and not the analytic derivative.

**Training-noise injection.** With `SCConfig.noise` set, layers add the
analytic counting noise of a length-$N$ stream to their output during
training, by reparameterization with a detached standard deviation:

$$y = v + \sigma(v)\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1),
\qquad \sigma \text{ detached},$$

so the gradient flows through the value, not through the noise magnitude. The
variance follows the accumulator (sections 3 and 4): for MUX, the
single-stream counting variance of the output value,
$\sigma^2 = (1 - v^2)/N$ bipolar, $v(1-v)/N$ unipolar
(`accuracy.sc_noise_std`); for APC, the summed per-term variances over $k^2$,
$\sigma^2 = \frac{1}{k^2 N} \sum_j (1 - v_j^2)$. After descaling by $k$ the
noise is $k$ times larger, which is precisely the signal-to-noise cost of deep
fan-in — this is how the optimizer feels finite stream length.

## 8. The dual-path invariant

scgrad maintains two implementations of every circuit. The approximate path
computes closed-form expectations and is differentiable. The exact path
(`eval_exact.py`) generates real bitstreams from `hardware.py`, pushes them
through real gate logic — XNOR or AND per bit, a MUX select stream or an APC
count per accumulation — and counts the output bits back to a value. Nothing
in it is differentiable and nothing in it samples from the approximate path;
both paths share only the layer port ids, so the randomness-budget structure
is identical.

The invariant that holds the library together:

$$\mathrm{exact}_f(x, N) \longrightarrow \mathrm{approx}_f(x)
\quad \text{as } N \to \infty,$$

for every supported circuit $f$ — each stage's counted estimate converges to
its expectation, by the law of large numbers for pseudo-random sources and by
equidistribution for low-discrepancy ones. The convergence test
(`tests/test_dual_path.py`) checks $\,\mathrm{mean}\,|\mathrm{exact} -
\mathrm{approx}| \le \tau(N, d)\,$ with the analytic tolerance of section 4
across a grid of stream lengths. If the two paths ever silently diverge, the
training path is optimizing a fiction; the test exists so that failure is loud.
It is the library's conscience.

## 9. The energy-based-model bridge (Phase 2)

The same `SCNumber` primitives extend to energy-based models. An Ising layer
(`ebm.py`) defines an energy over bipolar spins $s \in \{-1, +1\}^n$,

$$E(s) = -\tfrac{1}{2}\, s^\top J s - h^\top s,$$

with the coupling matrix $J$ held symmetric and zero-diagonal (constructed
from a raw parameter as $\tfrac{1}{2}(R + R^\top)$ minus its diagonal, so the
gradient respects the constraint instead of fighting it) and the field $h$ a
free parameter. The Boltzmann distribution at inverse temperature $\beta$ is
$\pi(s) \propto e^{-\beta E(s)}$.

**Gibbs sampling on bitstreams.** Single-site Gibbs flips spin $i$ to $+1$
with the conditional probability

$$p(s_i = +1 \mid s_{\neq i}) = \sigma\!\big(2\beta\,(J_i \cdot s + h_i)\big),$$

where $\sigma$ is the logistic function. This is exactly an SNG: a comparator
bit $\mathbb{1}[r < p]$ against a `hardware.py` uniform $r$ is a Bernoulli draw
with probability $p$, so each spin update *is* one stochastic-computing bit
whose probability is set by the local field. This is the p-bit picture, and the
randomness obeys the same seeding, reproducibility, and `n_rngs` budget
semantics as every other stream in the library. A systematic scan over all
sites, vectorized across chains, leaves $\pi$ invariant (each conditional
update satisfies detailed balance for $\pi$, and the composition of
$\pi$-invariant kernels is $\pi$-invariant), so the chain converges to the
Boltzmann distribution; `tests/test_ebm.py` checks the empirical distribution
against the analytic $\pi$ on a four-spin system within total-variation 0.08.

**Contrastive divergence.** Maximum-likelihood training of an EBM needs the
gradient $\partial_\theta \mathbb{E}_{\text{data}}[E] -
\partial_\theta \mathbb{E}_{\pi}[E]$, whose second term requires samples from
$\pi$. CD-$k$ (Hinton 2002) approximates $\mathbb{E}_\pi$ by $k$ Gibbs steps
started from the data. `contrastive_divergence` returns the scalar
$\mathbb{E}_{\text{data}}[E] - \mathbb{E}_{\text{neg}}[E]$ with the negative
samples detached, so its gradient with respect to $\theta$ is precisely the
CD-$k$ gradient. Song & Kingma (2021) survey the wider family of EBM training
methods this sits in.

**Chain independence.** Distinct chains, and distinct sites within a chain,
draw from distinct generator ids; chains sharing a generator would coalesce
(identical randomness plus identical kernels evolve identically), so ensemble
statistics require independent streams. A finite `n_rngs` correlating chains is
then a faithful model of a shared hardware RNG, not a defect.

**Honest positioning.** This is a software prototyping substrate, not a
hardware claim. The adjacent tools on the energy-based-sampling frontier —
THRML (Extropic) and thermox (Normal Computing) — are JAX and operate on spins
or Ornstein–Uhlenbeck processes close to a hardware story; scgrad's
contribution is the PyTorch-native, Gaines-lineage *bitstream* counterpart that
reuses the SC multiply/add/sample primitives directly. Where Extropic's
system-level energy figures (on the order of a $10^4\times$ projection) are
mentioned, they are simulation projections, not measured silicon, and are
labelled as such.

## References

- B. R. Gaines, "Stochastic computing," Spring Joint Computer Conference, 1967.
- B. R. Gaines, "Foundations of stochastic computing systems" (the XNOR
  relation $p^* = p_a p_b + (1-p_a)(1-p_b)$), IEEE note, 1968.
- B. R. Gaines, "Stochastic computing systems," Advances in Information
  Systems Science, vol. 2, 1969.
- A. Alaghi and J. P. Hayes, "Exploiting correlation in stochastic circuit
  design," ICCD 2013.
- A. Alaghi, P. Ting, V. Lee, J. P. Hayes, "Accuracy and correlation in
  stochastic computing," ch. 4 of Gross & Gaudet (eds.), *Stochastic
  Computing: Techniques and Applications*, Springer, 2019.
- M. Hsiao, J. Anderson, Y. Hara-Azumi, "Generating stochastic bitstreams,"
  in Gross & Gaudet (2019).
- M. Riedel, "Deterministic approaches to bitstream computing," in Gross &
  Gaudet (2019); with the Najafi/Riedel low-discrepancy line of work.
- D. Wu et al., "uGEMM: Unary computing architecture for GEMM applications,"
  ISCA 2020 (scaled-add error bounds; APC practice).
- A. Ren et al., "SC-DCNN: Highly-scalable deep convolutional neural network
  using stochastic computing," ASPLOS 2017 (APC practice in accelerators).
- Y. Bengio, N. Léonard, A. Courville, "Estimating or propagating gradients
  through stochastic neurons for conditional computation," arXiv:1308.3432,
  2013 (straight-through estimator).
