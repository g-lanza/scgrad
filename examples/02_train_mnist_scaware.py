"""Train a small SC-aware MNIST network and evaluate it on real bitstreams.

A compact version of the thesis benchmark: one SC model trained through
scgrad's differentiable SC forward (counting noise plus correlation
penalty), with per-layer output-gain calibration so the physical values
keep their dynamic range, then scored on the bit-accurate exact path.
Uses a training subset so it finishes in a couple of minutes on CPU; the
full comparison against float-then-map lives in
benchmarks/mnist_scaware_vs_float.py.

Run: uv run python examples/02_train_mnist_scaware.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from scgrad import (
    CorrelationTracker,
    SCConfig,
    SCLinear,
    SCNumber,
    SCReLU,
    correlation_loss,
    evaluate_exact,
    evaluate_float,
)

TRAIN_IMAGES = 12_800
EVAL_IMAGES = 500
LAMBDA_CORR = 0.2

config = SCConfig(
    encoding="bipolar",
    length=256,
    source="sobol",
    seed=5,
    noise=True,
    accumulator="apc",
)

tfm = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda t: t.reshape(-1) * 2.0 - 1.0)]
)
train = Subset(
    datasets.MNIST("data", train=True, download=True, transform=tfm), list(range(TRAIN_IMAGES))
)
test = Subset(
    datasets.MNIST("data", train=False, download=True, transform=tfm), list(range(EVAL_IMAGES))
)
train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=100, shuffle=False)

torch.manual_seed(0)
model = nn.Sequential(
    SCLinear(784, 64, config=config),
    SCReLU(),
    SCLinear(64, 10, config=config),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def calibrate_gains() -> None:
    """Set each SC layer's output gain from its activation range.

    Counteracts the 1/fan_in attenuation of scaled accumulation so the
    physical logits keep their dynamic range; the same calibration the
    deployed circuit would carry.
    """
    model.eval()
    x = next(iter(train_loader))[0]
    with torch.no_grad():
        cur: object = x
        for layer in model:
            if isinstance(layer, SCLinear):
                layer.output_gain = 1.0
                probe = layer(cur)
                assert isinstance(probe, SCNumber)
                q = float(torch.quantile(probe.value.abs().flatten(), 0.995).item())
                layer.output_gain = max(1.0, 0.95 / max(q, 1e-9))
                cur = layer(cur)
            else:
                cur = layer(cur)


for epoch in range(2):
    model.train()
    for step, (x, y) in enumerate(train_loader):
        with CorrelationTracker() as tracker:
            out = model(x)
        assert isinstance(out, SCNumber)
        logits = out.value / out.scale
        task = nn.functional.cross_entropy(logits, y)
        corr = correlation_loss(tracker)
        loss = task + LAMBDA_CORR * corr
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"epoch {epoch} step {step:3d}  task {task.item():.4f}  corr {corr.item():.4f}")
    calibrate_gains()

float_acc = evaluate_float(model, test_loader)["accuracy"]
print(f"\nfloat-forward accuracy ({EVAL_IMAGES} test images): {float_acc:.4f}")

for length in (256, 1024):
    eval_cfg = SCConfig(
        encoding="bipolar", length=length, source="sobol", seed=5, accumulator="apc"
    )
    exact_acc = evaluate_exact(model, test_loader, eval_cfg)["accuracy"]
    print(f"exact-path accuracy at N={length}: {exact_acc:.4f}")
