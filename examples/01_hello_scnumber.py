"""First contact with SCNumber: encode, multiply, add, and check the hardware.

Walks the core ideas in a few lines: values become bit probabilities,
multiplication is one gate, addition is a scaled average with a real
scale factor, and the bit-accurate path verifies the closed forms on
actual bitstreams.

Run: uv run python examples/01_hello_scnumber.py
"""

import torch

from scgrad import SCConfig, decode, encode, sc_add, sc_add_tree, sc_mul
from scgrad.hardware import make_source

config = SCConfig(encoding="bipolar", length=2048, source="sobol", seed=1)

a = encode(torch.tensor([0.6]), config)
b = encode(torch.tensor([-0.4]), config)
print(f"a = {a}")
print(f"b = {b}")

# Multiply: one XNOR gate per bit. Closed form in value space: v = v_a * v_b.
prod = sc_mul(a, b)
print(f"a*b   value = {prod.value.item():+.4f}  (float product {0.6 * -0.4:+.4f})")

# Add: a MUX computes the scaled average (a+b)/2, not the sum. The scale
# factor is tracked, and decode(descale=True) divides it back out.
total = sc_add(a, b)
print(
    f"a+b   physical = {total.value.item():+.4f}  scale = {total.scale:g}  "
    f"decoded = {decode(total).item():+.4f}"
)

# k-way accumulation: uniform MUX, physical output is the mean of the terms.
terms = [encode(torch.tensor([v]), config) for v in (0.5, -0.2, 0.3, 0.1)]
acc = sc_add_tree(terms)
print(
    f"tree  physical = {acc.value.item():+.4f}  scale = {acc.scale:g}  "
    f"decoded = {decode(acc).item():+.4f}  (true sum {0.5 - 0.2 + 0.3 + 0.1:+.4f})"
)

# The hardware truth: generate the real bitstreams and gate them.
source = make_source(config)
bits_a = source.bits(a.probabilities(), config.length, a.corr_id)
bits_b = source.bits(b.probabilities(), config.length, b.corr_id)
xnor = ~(bits_a ^ bits_b)
v_hat = 2.0 * xnor.double().mean().item() - 1.0
print(
    f"exact XNOR of {config.length} real bits: {v_hat:+.4f}  (closed form {prod.value.item():+.4f})"
)
