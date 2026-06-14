# scgrad — Complete Guide

> Differentiable stochastic-computing primitives for PyTorch.
> Train neural networks natively SC-aware.
>
> **github.com/g-lanza/scgrad** · MIT License · Python 3.11+ · PyTorch 2.2+

---

## 1. What Is scgrad?

scgrad is a Python library that lets you train neural networks designed to run on stochastic computing (SC) hardware — a type of chip that uses far less power than normal processors.

### The Problem It Solves

Normal chips represent numbers with 32 bits of precision. SC hardware takes a different approach:

- Numbers are represented as streams of random 1s and 0s
- A value of 0.7 means roughly 70% of the bits in the stream are 1s
- Multiplication becomes a single logic gate (XNOR) instead of a complex circuit
- The result: drastically less power, less heat, and smaller chips

The standard practice was to train a network normally in float32, then try to squeeze it onto SC hardware afterward — and accuracy would tank. scgrad trains the network while it already understands the hardware's constraints, so the optimizer learns around them from the start.

### The Headline Result

Tested on MNIST handwritten digit recognition, bit-accurate evaluation on real bitstreams:

| Method | N=256 (short streams) | N=1024 (long streams) |
|---|---|---|
| Standard (float-then-map) | 47% accuracy | 94% accuracy |
| scgrad (SC-aware training) | **84% accuracy** | 87% accuracy |

At short stream lengths — where SC hardware is actually fast and cheap — scgrad nearly doubles accuracy.

---

## 2. Use Cases

scgrad is for researchers and chip designers building AI on extremely low-power hardware:

- **Hearing aids** — noise cancellation on a chip the size of a grain of rice, running on a small battery for a week
- **Medical implants** — sensors inside the body that classify signals without a charging cable
- **Wildlife trackers** — glued to a bird or insect, running for months on a coin cell
- **Smart dust** — thousands of tiny sensors in a field or building with no battery replacement
- **Wearables** — a skin patch doing continuous health monitoring

All of these need pattern recognition but cannot run a normal chip — too hot, too power-hungry, too large. SC hardware does the math with a fraction of the transistors and watts. scgrad makes the networks accurate enough to be useful on that hardware.

---

## 3. From Library to Real Chip

**Step 1 — Train with scgrad** *(what this library does)*
Use scgrad's layers instead of standard PyTorch. The network learns to work well despite SC hardware's noise and limitations. Output: a file of trained weights.

**Step 2 — Export the weights**
Save the weights to a file. They're just numbers between -1 and 1.

**Step 3 — Describe the circuit in hardware**
An engineer writes the weights in VHDL or Verilog. Each weight becomes a comparator threshold. Each multiply becomes an XNOR gate. Each addition becomes a multiplexer.

**Step 4 — Simulate**
Simulate the gate-level circuit on a regular computer to verify correctness. scgrad's exact path already does a version of this.

**Step 5 — FPGA prototype**
Flash the gate design onto an FPGA (a reprogrammable chip) and validate it in real hardware before committing to a custom chip.

**Step 6 — Tape out**
Send the design to a chip fabricator (TSMC, GlobalFoundries, etc.) to manufacture. Expensive upfront; resulting chips cost pennies at scale and use microwatts of power.

scgrad provides Step 1 — the part that was previously missing. Steps 3–6 use existing hardware engineering workflows.

---

## 4. Setup

### Requirements

- Windows 10/11, macOS, or Linux
- ~500 MB disk space (PyTorch + MNIST data)

### Install uv

uv manages Python and all dependencies automatically.

**Windows (PowerShell):**
```
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Mac / Linux:**
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, close and reopen your terminal. Verify:
```
uv --version
```

### Install Git

**Windows:** Download from [git-scm.com/download/win](https://git-scm.com/download/win)

**Mac:** Run `git --version` — macOS will offer to install it if missing.

**Linux:** `sudo apt install git`

### Clone and install

```bash
git clone https://github.com/g-lanza/scgrad.git
cd scgrad
uv sync
```

Optional — also install the terminal GUI:
```bash
uv sync --extra gui
```

### Verify

```bash
uv run python -c "import scgrad; print('scgrad is working')"
```

---

## 5. What You Can Run

### Main benchmark

Trains two MNIST networks and compares them on real SC bitstreams:

```bash
uv run python benchmarks/mnist_scaware_vs_float.py --quick
```

Remove `--quick` for the full run (~2.5 minutes on CPU). Results are printed to terminal and saved to `docs/RESULTS.md`.

### Test suite

```bash
uv run pytest
```

### Quick API example

```python
import torch
from torch import nn
from scgrad import SCConfig, SCLinear, SCReLU

config = SCConfig(encoding="bipolar", length=256, noise=True, accumulator="apc")
model = nn.Sequential(
    SCLinear(784, 128, config=config),
    SCReLU(),
    SCLinear(128, 10, config=config),
)
out = model(torch.rand(32, 784) * 2 - 1)
loss = nn.functional.cross_entropy(out.value / out.scale, torch.randint(0, 10, (32,)))
loss.backward()
print("loss:", loss.item())
```

### Terminal GUI

Requires `uv sync --extra gui`. Shows live bitstream rasters, correlation matrix, error curves, and loss decomposition during training:

```bash
uv run scgrad-gui
```

---

## 6. What the Tests Prove

### test_dual_path.py — Does the fast math match real hardware?

The library has two computation paths:

- **Fast path** — a differentiable approximation used during training
- **Exact path** — a slow simulation that generates real bitstreams and runs them through real gate logic

These tests verify both paths agree, and the gap shrinks as stream length grows.

| Test | What it proves |
|---|---|
| Single layer | Fast path vs real bitstreams stay within the predicted error bound. Error halves as stream length grows 64x. |
| Two-layer network | Same check through a full network with activation function. |
| Convolution layer | Same check for image-processing layers. |
| LFSR source | The alternative random number generator also stays accurate. |
| Scale tracking | Both paths agree on the scale factor bitwise exactly. |
| Shared generator | When two streams share one RNG, multiplication silently breaks. The exact path reports the true (wrong) answer rather than hiding it. |

### test_correlation.py — Does the corruption detector work?

When two bitstreams share a random source, their math goes wrong. The library detects and penalizes this during training.

| Test | What it proves |
|---|---|
| Identical streams = 1.0 | Detector correctly reads perfect correlation. |
| Independent streams ≈ 0 | No false correlation on separate generators. |
| Constant streams | Edge case: all-1s or all-0s streams return 0 (no variance to correlate). |
| Opposite streams = -1.0 | Perfect anti-correlation detected correctly. |
| Tracker logs multiplications | Every multiply is recorded with the correct stream IDs. |
| Collision detection | Shared-generator pairs flagged; independent pairs not falsely flagged. |
| Penalty math is correct | Formula for shared-generator error matches known exact answers. |
| **Penalty predicts real error** | **The training penalty rank-correlates with actual hardware error (>0.8). If this fails, the training signal is a lie.** |
| No collision = zero penalty | When nothing shares a generator, loss is exactly 0. No false alarms. |
| Penalty has correct value | For 0.5 × -0.5 with shared RNG, penalty is exactly 0.25 (to 6 decimal places). |
| Gradients flow | PyTorch can backpropagate through the penalty to the weights. |

---

## 7. Glossary

| Term | Plain English |
|---|---|
| Stochastic computing (SC) | Representing numbers as streams of random bits. 0.7 = ~70% ones. |
| Stream length N | How many bits in the stream. Longer = more accurate but slower hardware. |
| Bipolar encoding | Maps values between -1 and 1 into bit probabilities between 0 and 1. |
| XNOR gate | The single logic gate that performs multiplication in SC. |
| APC | Accumulative parallel counter — how multiple SC values are added. Used in real published SC chips. |
| float-then-map | The old standard: train in float32, then post-hoc convert to SC. scgrad improves on this. |
| Correlation / SCC | How much two bitstreams depend on each other. Shared RNG causes correlation, which breaks multiplication. |
| Correlation penalty | A loss term that penalizes shared-generator multiplies during training. |
| Exact path | Bit-accurate simulation using real bitstreams and real gates. The honest measurement. |
| Fast path | The differentiable approximation used during training. Converges to exact path as N grows. |
| n_rngs | How many physical random number generators the hardware has. Fewer = more sharing = more error. |

---

## 8. Troubleshooting

| Error | Fix |
|---|---|
| `uv: command not found` | Close and reopen your terminal after installing uv. |
| `No pyproject.toml found` | Wrong directory. Run `cd scgrad` first. |
| `ModuleNotFoundError: scgrad` | Use `uv run python` instead of just `python`. |
| Benchmark seems stuck | Downloading MNIST dataset (~11 MB) or training. Wait a few minutes. |
| `sh is not recognized` (Windows) | You are in Command Prompt. Open PowerShell instead. |

---

## 9. Repository Structure

```
scgrad/
  benchmarks/        — runnable experiments (mnist_scaware_vs_float.py is the main one)
  docs/              — theory.md, design_notes.md, RESULTS.md
  src/scgrad/        — the library (encoding, ops, layers, hardware, correlation, eval_exact, accuracy, gui)
  tests/             — test_dual_path.py, test_correlation.py
  pyproject.toml     — project config (Python 3.11+, torch 2.2+, numpy 1.26+)
```
