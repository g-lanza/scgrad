"""The bit-accurate reference path: real bitstreams, real gates, counted back.

This is the honest hardware truth scgrad is held to. exact_forward
replays an SC model's circuit on real bitstreams from hardware.py: each
layer generates streams at its ports, multiplies them bit by bit through
XNOR (bipolar) or AND (unipolar), accumulates through a uniform MUX
select stream, and counts the output bits back to a value. Nothing here
is differentiable and nothing samples from the approximate path.

Vectorization: bits are processed in time chunks, and the per-chunk
XNOR/AND counting is expressed as matmuls over the time axis
(count of coinciding ones = bits_a @ bits_b.T), so the work is batched
across whole tensors with no per-element Python loops. Counts stay below
2^24, exact in float32.

Models must be nn.Sequential pipelines of SCLinear / SCConv2d /
SCReLU / nn.ReLU / SCFlatten / nn.Flatten. The approximate and exact
paths share the layer port ids, so the randomness-budget structure
(SCConfig.n_rngs) is identical in both. Use SCFlatten (not nn.Flatten)
between a conv stage and a linear stage so the approximate path can carry
an SCNumber across the reshape; the exact path treats both identically.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from scgrad.encoding import SCConfig, SCEncodingError, SCNumber, value_to_probability
from scgrad.hardware import BitstreamSource, make_source
from scgrad.layers import SCConv2d, SCFlatten, SCLinear, SCReLU

_CHUNK = 256


@dataclass
class _ExactState:
    """Physical value flowing through the exact circuit."""

    value: Tensor
    scale: float
    corr_id: int


def _select_stream(config: SCConfig, corr_id: int, k: int, n: int) -> Tensor:
    """Deterministic uniform MUX select stream for a layer output."""
    gen = torch.Generator()
    seed = (config.seed if config.seed is not None else 0) * 1_000_003 + corr_id * 65_537 + k
    gen.manual_seed(seed % (2**63 - 1))
    return torch.randint(0, k, (n,), generator=gen)


def _mux_linear_counts(
    x_vals: Tensor,
    w_vals: Tensor,
    bias_vals: Tensor | None,
    config: SCConfig,
    x_id: int,
    w_id: int,
    b_id: int,
    out_id: int,
) -> Tensor:
    """Exact MUX-accumulated multiply for a linear stage.

    x_vals: (B, K) input values; w_vals: (O, K) weight values;
    bias_vals: (O,) pre-scaled bias values or None. Returns (B, O)
    counts of output ones over the full stream length N.

    At each time step t the MUX selects one term j_t; only the bits of
    the selected column are ever needed, so they are gathered straight
    from the comparator inequality bit = (r_t < p) against the source's
    uniform sequence instead of materializing whole streams. The
    per-chunk XNOR/AND counting is a matmul over the time axis
    (coinciding ones = x_bits @ w_bits.T, plus coinciding zeros for
    XNOR).
    """
    n = config.length
    in_features = x_vals.shape[-1]
    k_terms = in_features + (1 if bias_vals is not None else 0)
    source: BitstreamSource = make_source(config)
    select = _select_stream(config, out_id, k_terms, n)
    p_x = value_to_probability(x_vals.detach(), config.encoding).to(torch.float32)
    p_w = value_to_probability(w_vals.detach(), config.encoding).to(torch.float32)
    r_x = source.uniforms(n, x_id).to(torch.float32)
    r_w = source.uniforms(n, w_id).to(torch.float32)
    p_b: Tensor | None = None
    r_b: Tensor | None = None
    if bias_vals is not None:
        p_b = value_to_probability(bias_vals.detach(), config.encoding).to(torch.float32)
        r_b = source.uniforms(n, b_id).to(torch.float32)
    counts = torch.zeros(x_vals.shape[0], w_vals.shape[0], dtype=torch.float32)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        sel = select[start:stop]
        gate_mask = sel < in_features
        sel_gate = sel.clamp(max=in_features - 1)
        x_bits = (r_x[start:stop] < p_x[:, sel_gate]) & gate_mask
        w_bits = (r_w[start:stop] < p_w[:, sel_gate]) & gate_mask
        ones = x_bits.to(torch.float32) @ w_bits.to(torch.float32).t()
        if config.encoding == "bipolar":
            x_zero = (~x_bits) & gate_mask
            w_zero = (~w_bits) & gate_mask
            ones = ones + x_zero.to(torch.float32) @ w_zero.to(torch.float32).t()
        counts = counts + ones
        if p_b is not None and r_b is not None:
            bias_bits = (r_b[start:stop] < p_b.unsqueeze(-1)) & (sel == in_features)
            counts = counts + bias_bits.sum(dim=-1).to(torch.float32)
    return counts


def _apc_linear_value(
    x_vals: Tensor,
    w_vals: Tensor,
    bias_vals: Tensor | None,
    config: SCConfig,
    x_id: int,
    w_id: int,
    b_id: int,
) -> Tensor:
    """Exact APC-accumulated multiply: every product bit counted every clock.

    The accumulative parallel counter sums all k gate outputs each time
    step into a binary count (no selection, hence no selection noise),
    then normalizes by k in the binary domain. Counting XNOR/AND ones
    across both the term and time axes is a matmul over the flattened
    (term, time) axis, blocked over time chunks and batch rows to bound
    memory. Counts stay below 2^24 per block, exact in float32.
    """
    n = config.length
    k_in = x_vals.shape[-1]
    n_batch, n_out = x_vals.shape[0], w_vals.shape[0]
    source: BitstreamSource = make_source(config)
    p_x = value_to_probability(x_vals.detach(), config.encoding).to(torch.float32)
    p_w = value_to_probability(w_vals.detach(), config.encoding).to(torch.float32)
    r_x = source.uniforms(n, x_id).to(torch.float32)
    r_w = source.uniforms(n, w_id).to(torch.float32)
    ones = torch.zeros(n_batch, n_out, dtype=torch.float32)
    zeros = torch.zeros(n_batch, n_out, dtype=torch.float32)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        c = stop - start
        w_bits = r_w[start:stop] < p_w.unsqueeze(-1)
        wf = w_bits.reshape(n_out, k_in * c).to(torch.float32)
        wzf = (~w_bits).reshape(n_out, k_in * c).to(torch.float32)
        row_block = max(1, 2**24 // max(k_in * c, 1))
        for rs in range(0, n_batch, row_block):
            r_end = min(rs + row_block, n_batch)
            x_bits = r_x[start:stop] < p_x[rs:r_end].unsqueeze(-1)
            xf = x_bits.reshape(r_end - rs, k_in * c).to(torch.float32)
            ones[rs:r_end] += xf @ wf.t()
            if config.encoding == "bipolar":
                zeros[rs:r_end] += (1.0 - xf) @ wzf.t()
    is_bipolar = config.encoding == "bipolar"
    sum_v = 2.0 * (ones + zeros) / n - float(k_in) if is_bipolar else ones / n
    k_terms = k_in
    if bias_vals is not None:
        k_terms += 1
        p_b = value_to_probability(bias_vals.detach(), config.encoding).to(torch.float32)
        r_b = source.uniforms(n, b_id).to(torch.float32)
        count_b = (r_b < p_b.unsqueeze(-1)).sum(dim=-1).to(torch.float32)
        v_b = 2.0 * count_b / n - 1.0 if config.encoding == "bipolar" else count_b / n
        sum_v = sum_v + v_b
    return sum_v / k_terms


def _linear_stage_value(
    x_vals: Tensor,
    w_vals: Tensor,
    bias_vals: Tensor | None,
    config: SCConfig,
    x_id: int,
    w_id: int,
    b_id: int,
    out_id: int,
) -> Tensor:
    """Counted physical output value of one linear stage under the config's accumulator."""
    if config.accumulator == "apc":
        return _apc_linear_value(x_vals, w_vals, bias_vals, config, x_id, w_id, b_id)
    counts = _mux_linear_counts(x_vals, w_vals, bias_vals, config, x_id, w_id, b_id, out_id)
    p_hat = counts / config.length
    return p_hat if config.encoding == "unipolar" else 2.0 * p_hat - 1.0


def _apply_gain(value: Tensor, gain: float, config: SCConfig) -> Tensor:
    """Binary-domain output gain with range saturation (mirrors layers)."""
    if gain == 1.0:
        return value
    lo, hi = _enc_range(config)
    return (value * gain).clamp(lo, hi)


def _exact_linear(state: _ExactState, layer: SCLinear, config: SCConfig) -> _ExactState:
    w = layer.weight.detach().clamp(*_enc_range(config))
    bias = None
    if layer.bias is not None:
        bias = (state.scale * layer.bias.detach()).clamp(*_enc_range(config))
    value = _linear_stage_value(
        state.value,
        w,
        bias,
        config,
        state.corr_id,
        layer.weight_corr_id,
        layer.bias_corr_id,
        layer.output_corr_id,
    )
    value = _apply_gain(value, layer.output_gain, config)
    return _ExactState(value, state.scale * layer.output_gain / layer.fan_in, layer.output_corr_id)


def _exact_conv(state: _ExactState, layer: SCConv2d, config: SCConfig) -> _ExactState:
    batch, _, h, w_in = state.value.shape
    out_h, out_w = layer.output_size(h, w_in)
    patches = nn.functional.unfold(
        state.value,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
    )
    n_loc = patches.shape[-1]
    flat = patches.transpose(1, 2).reshape(batch * n_loc, -1)
    w_flat = layer.weight.detach().reshape(layer.out_channels, -1).clamp(*_enc_range(config))
    bias = None
    if layer.bias is not None:
        bias = (state.scale * layer.bias.detach()).clamp(*_enc_range(config))
    value = _linear_stage_value(
        flat,
        w_flat,
        bias,
        config,
        state.corr_id,
        layer.weight_corr_id,
        layer.bias_corr_id,
        layer.output_corr_id,
    )
    value = value.reshape(batch, n_loc, layer.out_channels).transpose(1, 2)
    value = value.reshape(batch, layer.out_channels, out_h, out_w)
    value = _apply_gain(value, layer.output_gain, config)
    return _ExactState(value, state.scale * layer.output_gain / layer.fan_in, layer.output_corr_id)


def _enc_range(config: SCConfig) -> tuple[float, float]:
    return (0.0, 1.0) if config.encoding == "unipolar" else (-1.0, 1.0)


def exact_forward(model: nn.Sequential, x: Tensor, config: SCConfig) -> SCNumber:
    """Run the model on real bitstreams and real gate logic; return the output.

    The returned SCNumber's value is the counted physical output and its
    scale is the accumulated MUX factor, matching the approximate path's
    bookkeeping exactly. Padding bits in the unfold path are encoded as
    value zero, the same fill the approximate path uses.
    """
    lo, hi = _enc_range(config)
    first = _first_sc_layer(model)
    sc_layers = [m for m in model if isinstance(m, (SCLinear, SCConv2d))]
    for m in sc_layers[:-1]:
        if m.decode_output:
            raise SCEncodingError(
                "decode_output on a non-final SC layer breaks the physical "
                "correspondence between paths; decode only at the output"
            )
    state = _ExactState(x.detach().clamp(lo, hi), 1.0, first.input_corr_id)
    for layer in model:
        if isinstance(layer, SCLinear):
            state = _exact_linear(state, layer, config)
        elif isinstance(layer, SCConv2d):
            state = _exact_conv(state, layer, config)
        elif isinstance(layer, (SCReLU, nn.ReLU)):
            state = _ExactState(torch.relu(state.value), state.scale, state.corr_id)
        elif isinstance(layer, SCFlatten):
            state = _ExactState(state.value.flatten(layer.start_dim), state.scale, state.corr_id)
        elif isinstance(layer, nn.Flatten):
            state = _ExactState(layer(state.value), state.scale, state.corr_id)
        else:
            raise SCEncodingError(f"exact_forward cannot replay layer type {type(layer).__name__}")
    return SCNumber(state.value, config, scale=state.scale, corr_id=state.corr_id)


def _first_sc_layer(model: nn.Sequential) -> SCLinear | SCConv2d:
    for layer in model:
        if isinstance(layer, (SCLinear, SCConv2d)):
            return layer
    raise SCEncodingError("model contains no SC layers")


def evaluate_exact(
    model: nn.Sequential,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    config: SCConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Classification accuracy on the bit-accurate path: the hardware truth.

    Argmax is taken on the physical output values; all logits of one
    forward share a single scale, so the descaled argmax is identical.
    """
    correct = 0
    total = 0
    for i, (x, y) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        out = exact_forward(model, x, config)
        pred = out.value.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return {"accuracy": correct / max(total, 1), "n": float(total)}


def evaluate_float(
    model: nn.Module,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    max_batches: int | None = None,
) -> dict[str, float]:
    """Classification accuracy of the model's float forward: the upper bound."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            out = model(x)
            logits = out.value / out.scale if isinstance(out, SCNumber) else out
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    if was_training:
        model.train()
    return {"accuracy": correct / max(total, 1), "n": float(total)}
