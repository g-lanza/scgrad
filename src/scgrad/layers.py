"""Drop-in PyTorch layers built from the SC primitives.

Both layers compute, in closed form, exactly what the elementwise
primitive graph (sc_mul per product, sc_add_tree over the fan-in)
computes in expectation: the products collapse to a matmul and the
uniform MUX accumulation to a division by the fan-in k. The collapsed
form is one tensor op, GPU-friendly, and is tested for equivalence
against the elementwise primitives in tests/test_layers.py.

Scale behavior (stated per the architecture contract): with input scale
s_in and fan-in k (in_features, plus one for the bias term when
present), the output value is the physical MUX result
(x @ W.T + s_in * b) / k and the output scale is s_in / k. The encoded
bias is pre-scaled by s_in so every MUX term carries the same factor,
which keeps decode(descale=True) an exact recovery of x @ W.T + b; this
mirrors programming a hardware bias register in the incoming scale.

Correlation identities: each layer owns persistent port ids (input,
weight, bias, output), allocated at construction, the way SNGs are fixed
silicon at each port. Layer outputs are regenerated streams (count, then
re-encode through the output port SNG), the standard decorrelation
practice; partial correlation propagation through op outputs is
deliberately not modeled in v0.1 (see docs/design_notes.md).

Training noise: with config.noise set, layers inject the analytic
counting noise of a length-N stream (accuracy.sc_noise_std) into their
output during training, reparameterized with a detached standard
deviation. This is how the optimizer feels finite stream length: after
descaling, the noise is k times larger, which is precisely the SC
signal-to-noise cost of deep fan-in.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from scgrad.accuracy import sc_noise_std
from scgrad.correlation import record_multiply
from scgrad.encoding import (
    SCConfig,
    SCEncodingError,
    SCNumber,
    clamp_ste,
    fresh_corr_id,
)


def _apply_noise(
    value: Tensor, config: SCConfig, training: bool, term_var_sum: Tensor | None = None
) -> Tensor:
    """Inject analytic SC counting noise during training (reparameterized).

    MUX accumulation: the output is one counted stream, so the noise is
    the single-stream counting std of the output value. APC accumulation:
    every term stream is counted exactly, so the output variance is the
    sum of per-term counting variances divided by k^2; the caller passes
    term_var_sum = sum_j var_numerator_j (bipolar: 1 - v_j^2; unipolar:
    p_j(1 - p_j)) already divided by k^2, and this function scales by
    1/length. The std is detached: the gradient flows through the value,
    not through the noise magnitude.
    """
    if not (config.noise and training):
        return value
    if config.accumulator == "mux" or term_var_sum is None:
        std = sc_noise_std(value.detach(), config.length, config.encoding)
    else:
        std = torch.sqrt(torch.clamp(term_var_sum.detach(), min=0.0) / config.length)
    noisy = value + std * torch.randn_like(value)
    return clamp_ste(noisy, config.encoding)


def _apc_term_var_sum(
    x_val: Tensor, w: Tensor, b_enc: Tensor | None, k: int, cfg: SCConfig
) -> Tensor:
    """Per-output sum of APC term counting-variance numerators over k^2 (linear)."""
    with torch.no_grad():
        if cfg.encoding == "bipolar":
            s = x_val.shape[-1] - (x_val**2) @ (w**2).t()
            if b_enc is not None:
                s = s + (1.0 - b_enc**2)
        else:
            s = x_val @ w.t() - (x_val**2) @ (w**2).t()
            if b_enc is not None:
                s = s + b_enc * (1.0 - b_enc)
    return s / (k * k)


def _apc_term_var_sum_conv(
    patches: Tensor, w_flat: Tensor, b_enc: Tensor | None, k: int, cfg: SCConfig
) -> Tensor:
    """Per-output sum of APC term counting-variance numerators over k^2 (conv)."""
    with torch.no_grad():
        if cfg.encoding == "bipolar":
            s = patches.shape[1] - torch.einsum("bkl,ok->bol", patches**2, w_flat**2)
            if b_enc is not None:
                s = s + (1.0 - b_enc**2).reshape(1, -1, 1)
        else:
            s = torch.einsum("bkl,ok->bol", patches, w_flat) - torch.einsum(
                "bkl,ok->bol", patches**2, w_flat**2
            )
            if b_enc is not None:
                s = s + (b_enc * (1.0 - b_enc)).reshape(1, -1, 1)
    return s / (k * k)


class SCLinear(nn.Module):
    """SC linear layer: XNOR/AND multiplies accumulated by a uniform MUX.

    Mirrors nn.Linear: float weight/bias Parameters trained normally,
    clamped into the encoding range (straight-through) when encoded each
    forward. forward is the approximate (training) path; the exact path
    in eval_exact.py replays the same circuit on real bitstreams using
    this layer's port ids. Returns an SCNumber, or the descaled tensor
    when decode_output is set.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: SCConfig | None = None,
        decode_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config if config is not None else SCConfig()
        self.decode_output = decode_output
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias: nn.Parameter | None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()
        self.input_corr_id = fresh_corr_id()
        self.weight_corr_id = fresh_corr_id()
        self.bias_corr_id = fresh_corr_id()
        self.output_corr_id = fresh_corr_id()
        # Binary-domain gain register applied to the counted output before
        # re-encoding (a fixed-point multiply on hardware). Counteracts the
        # 1/fan_in dynamic-range loss of scaled accumulation; values that
        # leave the encoding range saturate, in both paths. Standard
        # practice in SC accelerators; calibrate per layer on real
        # activations (see benchmarks/mnist_scaware_vs_float.py).
        self.output_gain: float = 1.0

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def fan_in(self) -> int:
        """MUX fan-in k: in_features plus one for the bias term."""
        return self.in_features + (1 if self.bias is not None else 0)

    def forward(self, x: Tensor | SCNumber) -> SCNumber | Tensor:
        cfg = self.config
        if isinstance(x, SCNumber):
            if x.config.encoding != cfg.encoding:
                raise SCEncodingError(
                    f"layer encoding {cfg.encoding} got input encoding {x.config.encoding}"
                )
            x_val, x_scale, x_id = x.value, x.scale, x.corr_id
        else:
            x_val = clamp_ste(x, cfg.encoding)
            x_scale, x_id = 1.0, self.input_corr_id
        w = clamp_ste(self.weight, cfg.encoding)
        record_multiply(
            SCNumber(x_val.unsqueeze(-2), cfg, scale=x_scale, corr_id=x_id),
            SCNumber(w, cfg, scale=1.0, corr_id=self.weight_corr_id),
        )
        k = self.fan_in
        out_val = x_val @ w.t()
        b_enc: Tensor | None = None
        if self.bias is not None:
            # The bias register is programmed in the incoming scale, then
            # encoded: clamp AFTER scaling, matching the exact path.
            b_enc = clamp_ste(x_scale * self.bias, cfg.encoding)
            out_val = out_val + b_enc
        out_val = out_val / k
        term_var_sum: Tensor | None = None
        if cfg.accumulator == "apc" and cfg.noise and self.training:
            term_var_sum = _apc_term_var_sum(x_val, w, b_enc, k, cfg)
        out_val = _apply_noise(out_val, cfg, self.training, term_var_sum)
        if self.output_gain != 1.0:
            out_val = clamp_ste(out_val * self.output_gain, cfg.encoding)
        out = SCNumber(
            out_val,
            cfg,
            scale=x_scale * self.output_gain / k,
            corr_id=self.output_corr_id,
        )
        if self.decode_output:
            from scgrad.encoding import decode

            return decode(out, descale=True)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, N={self.config.length}, "
            f"encoding={self.config.encoding}, scale=1/{self.fan_in}, "
            f"gain={self.output_gain:g}"
        )


class SCConv2d(nn.Module):
    """SC 2d convolution via im2col: the SCLinear circuit over each patch.

    Matches the nn.Conv2d signature for stride, padding, and dilation;
    grouped convolution is not part of the v0.1 circuit (groups must be
    1). MUX fan-in is in_channels * kernel_height * kernel_width, plus
    one for the bias term; scale behavior is identical to SCLinear with
    that fan-in.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        config: SCConfig | None = None,
        decode_output: bool = False,
    ) -> None:
        super().__init__()
        if groups != 1:
            raise SCEncodingError("SCConv2d supports groups=1 only")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.config = config if config is not None else SCConfig()
        self.decode_output = decode_output
        kh, kw = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias: nn.Parameter | None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()
        self.input_corr_id = fresh_corr_id()
        self.weight_corr_id = fresh_corr_id()
        self.bias_corr_id = fresh_corr_id()
        self.output_corr_id = fresh_corr_id()
        # Binary-domain gain register; see SCLinear.output_gain.
        self.output_gain: float = 1.0

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            kh, kw = self.kernel_size
            bound = 1.0 / math.sqrt(self.in_channels * kh * kw)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def fan_in(self) -> int:
        """MUX fan-in k: patch size plus one for the bias term."""
        kh, kw = self.kernel_size
        return self.in_channels * kh * kw + (1 if self.bias is not None else 0)

    def output_size(self, h: int, w: int) -> tuple[int, int]:
        """Spatial output size for an input of height h and width w."""
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        out_h = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return out_h, out_w

    def forward(self, x: Tensor | SCNumber) -> SCNumber | Tensor:
        cfg = self.config
        if isinstance(x, SCNumber):
            if x.config.encoding != cfg.encoding:
                raise SCEncodingError(
                    f"layer encoding {cfg.encoding} got input encoding {x.config.encoding}"
                )
            x_val, x_scale = x.value, x.scale
            x_id = x.corr_id
        else:
            x_val = clamp_ste(x, cfg.encoding)
            x_scale, x_id = 1.0, self.input_corr_id
        batch, _, h, w = x_val.shape
        out_h, out_w = self.output_size(h, w)
        patches = nn.functional.unfold(
            x_val,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        w_enc = clamp_ste(self.weight, cfg.encoding)
        w_flat = w_enc.reshape(self.out_channels, -1)
        record_multiply(
            SCNumber(patches.transpose(1, 2).unsqueeze(-2), cfg, scale=x_scale, corr_id=x_id),
            SCNumber(w_flat, cfg, scale=1.0, corr_id=self.weight_corr_id),
        )
        k = self.fan_in
        out_val = torch.einsum("bkl,ok->bol", patches, w_flat)
        b_enc: Tensor | None = None
        if self.bias is not None:
            # Bias register programmed in the incoming scale, then encoded
            # (clamp after scaling), matching the exact path.
            b_enc = clamp_ste(x_scale * self.bias, cfg.encoding)
            out_val = out_val + b_enc.reshape(1, -1, 1)
        out_val = out_val / k
        term_var_sum: Tensor | None = None
        if cfg.accumulator == "apc" and cfg.noise and self.training:
            term_var_sum = _apc_term_var_sum_conv(patches, w_flat, b_enc, k, cfg).reshape(
                batch, self.out_channels, out_h, out_w
            )
        out_val = out_val.reshape(batch, self.out_channels, out_h, out_w)
        out_val = _apply_noise(out_val, cfg, self.training, term_var_sum)
        if self.output_gain != 1.0:
            out_val = clamp_ste(out_val * self.output_gain, cfg.encoding)
        out = SCNumber(
            out_val,
            cfg,
            scale=x_scale * self.output_gain / k,
            corr_id=self.output_corr_id,
        )
        if self.decode_output:
            from scgrad.encoding import decode

            return decode(out, descale=True)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None}, "
            f"N={self.config.length}, encoding={self.config.encoding}, scale=1/{self.fan_in}, "
            f"gain={self.output_gain:g}"
        )


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(v, tuple):
        return v
    return (v, v)


def sc_relu(s: SCNumber) -> SCNumber:
    """ReLU between SC layers, applied in the decoded domain.

    ReLU is positively homogeneous, so applying it to the scaled physical
    value equals applying it to the descaled value and rescaling: the
    scale carries through unchanged and no decode/clamp round-trip is
    needed. The corr_id also carries through: ReLU is a digital-domain
    operation on the counted value, upstream of the producing layer's
    output SNG, so the stream identity it flows on is unchanged (the
    exact path does the same). Acceptable v0.1 activation treatment,
    documented in docs/design_notes.md; SC-native activation circuits
    are future work.
    """
    return SCNumber(torch.relu(s.value), s.config, scale=s.scale, corr_id=s.corr_id)


class SCReLU(nn.Module):
    """Module wrapper around sc_relu for use inside nn.Sequential."""

    def forward(self, x: SCNumber) -> SCNumber:
        return sc_relu(x)


def sc_flatten(s: SCNumber, start_dim: int = 1) -> SCNumber:
    """Flatten an SCNumber's value tensor from start_dim, preserving metadata.

    A reshape is wiring, not arithmetic: it relabels which physical wire
    carries which value and changes neither the values, the scale, nor
    the stream identity. The exact path flattens the raw value tensor
    identically (it carries no SCNumber across nn.Flatten), so the two
    paths agree.
    """
    return SCNumber(s.value.flatten(start_dim), s.config, scale=s.scale, corr_id=s.corr_id)


class SCFlatten(nn.Module):
    """SC-aware flatten between a convolutional stage and a linear stage.

    Use this in an nn.Sequential instead of nn.Flatten when the tensor in
    flight is an SCNumber (the approximate/training path). The exact path
    accepts a plain nn.Flatten because it threads raw tensors, so a model
    built for both paths places SCFlatten here and exact_forward treats it
    the same as nn.Flatten.
    """

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: SCNumber) -> SCNumber:
        return sc_flatten(x, self.start_dim)
