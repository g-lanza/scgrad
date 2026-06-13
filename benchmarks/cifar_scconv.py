"""Smoke-scale CIFAR-10 benchmark for SCConv2d: SC-aware vs float-then-map.

Not the thesis gate (that is the MNIST benchmark). This exercises the
convolutional layer end to end on real images and reports exact-path
accuracy, so SCConv2d is proven on a task rather than only in unit tests.
A small convolutional stem feeds a single SC linear classifier; training
is on a CIFAR subset to keep CPU runtime modest.

Run: uv run python benchmarks/cifar_scconv.py [--quick]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from scgrad.correlation import CorrelationTracker, correlation_loss
from scgrad.encoding import SCConfig, SCNumber
from scgrad.eval_exact import evaluate_exact, evaluate_float
from scgrad.layers import SCConv2d, SCFlatten, SCLinear, SCReLU

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

CONV_OUT = 12
FLAT = CONV_OUT * 14 * 14
LAMBDA_CORR = 0.2
EVAL_LENGTHS = (256, 1024)


def _loaders(eval_subset: int, train_subset: int) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t * 2.0 - 1.0)])
    train = datasets.CIFAR10(str(DATA_DIR), train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=tfm)
    return (
        DataLoader(Subset(train, list(range(train_subset))), batch_size=64, shuffle=True),
        DataLoader(Subset(test, list(range(eval_subset))), batch_size=50, shuffle=False),
    )


def _config(length: int, noise: bool) -> SCConfig:
    return SCConfig(
        encoding="bipolar", length=length, source="sobol", seed=13, noise=noise, accumulator="apc"
    )


def _build(config: SCConfig) -> nn.Sequential:
    return nn.Sequential(
        SCConv2d(3, CONV_OUT, 5, stride=2, config=config),
        SCReLU(),
        SCFlatten(),
        SCLinear(FLAT, 10, config=config),
    )


def calibrate_gains(model: nn.Sequential, loader: DataLoader, n_batches: int = 2) -> None:
    """Per-layer output-gain calibration for SCConv2d and SCLinear.

    Generalizes the MNIST benchmark's calibration to a mixed conv/linear
    pipeline: walk the sequence, and for each SC layer set the binary-
    domain gain from the 0.995 activation quantile. Applied identically
    to both contenders so it confers no method-specific advantage.
    """
    model.eval()
    xs = [x for i, (x, _) in enumerate(loader) if i < n_batches]
    batch = torch.cat(xs)
    with torch.no_grad():
        cur: Tensor | SCNumber = batch
        for layer in model:
            if isinstance(layer, SCConv2d | SCLinear):
                layer.output_gain = 1.0
                probe = layer(cur)
                assert isinstance(probe, SCNumber)
                q = float(torch.quantile(probe.value.abs().flatten(), 0.995).item())
                layer.output_gain = max(1.0, 0.95 / max(q, 1e-9))
                cur = layer(cur)
            else:
                cur = layer(cur)


def _train_float(loader: DataLoader, epochs: int) -> nn.Sequential:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, CONV_OUT, 5, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(FLAT, 10),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in loader:
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _map_float_to_sc(float_model: nn.Sequential, config: SCConfig) -> nn.Sequential:
    sc_model = _build(config)
    floats = [m for m in float_model if isinstance(m, nn.Conv2d | nn.Linear)]
    scs = [m for m in sc_model if isinstance(m, SCConv2d | SCLinear)]
    g = 1.0
    with torch.no_grad():
        for fl, sl in zip(floats, scs, strict=True):
            assert fl.bias is not None and sl.bias is not None
            f = max(1.0, float(fl.weight.abs().max().item()), float(fl.bias.abs().max().item()) / g)
            sl.weight.copy_(fl.weight.reshape(sl.weight.shape) / f)
            sl.bias.copy_(fl.bias / (g * f))
            g *= f
    return sc_model


def _train_sc_aware(loader: DataLoader, config: SCConfig, epochs: int) -> nn.Sequential:
    torch.manual_seed(0)
    model = _build(config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            with CorrelationTracker() as tracker:
                out = model(x)
            assert isinstance(out, SCNumber)
            logits = out.value / out.scale
            loss = nn.functional.cross_entropy(logits, y) + LAMBDA_CORR * correlation_loss(tracker)
            opt.zero_grad()
            loss.backward()
            opt.step()
        calibrate_gains(model, loader)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="1 epoch, 2000/200 images")
    args = parser.parse_args()
    epochs = 1 if args.quick else 2
    train_subset = 2000 if args.quick else 10_000
    eval_subset = 200 if args.quick else 500

    t0 = time.time()
    train_loader, test_loader = _loaders(eval_subset, train_subset)

    print("training float baseline...", flush=True)
    float_model = _train_float(train_loader, epochs)
    print(f"float32 accuracy: {evaluate_float(float_model, test_loader)['accuracy']:.4f}")

    print("training SC-aware SCConv2d model...", flush=True)
    sc_model = _train_sc_aware(train_loader, _config(EVAL_LENGTHS[0], noise=True), epochs)
    sc_float = evaluate_float(sc_model, test_loader)["accuracy"]
    print(f"SC-aware float-forward accuracy: {sc_float:.4f}")

    print(f"\n{'method':>16}  {'N':>5}  {'exact-path accuracy':>20}")
    for length in EVAL_LENGTHS:
        cfg = _config(length, noise=False)
        mapped = _map_float_to_sc(float_model, cfg)
        calibrate_gains(mapped, train_loader)
        acc_a = evaluate_exact(mapped, test_loader, cfg)["accuracy"]
        print(f"{'float-then-map':>16}  {length:5d}  {acc_a:20.4f}", flush=True)
        sc_eval = _build(cfg)
        sc_eval.load_state_dict(sc_model.state_dict())
        for s, d in zip(
            [m for m in sc_model if isinstance(m, SCConv2d | SCLinear)],
            [m for m in sc_eval if isinstance(m, SCConv2d | SCLinear)],
            strict=True,
        ):
            # Port stream ids are plain attributes, not state_dict entries;
            # copy them so the exact path replays the trained circuit.
            d.output_gain = s.output_gain
            d.input_corr_id = s.input_corr_id
            d.weight_corr_id = s.weight_corr_id
            d.bias_corr_id = s.bias_corr_id
            d.output_corr_id = s.output_corr_id
        acc_b = evaluate_exact(sc_eval, test_loader, cfg)["accuracy"]
        print(f"{'sc-aware':>16}  {length:5d}  {acc_b:20.4f}", flush=True)
    print(f"\nwall time: {time.time() - t0:.0f} s (CIFAR-10 subset, CPU, single seed)")


if __name__ == "__main__":
    main()
