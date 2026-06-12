"""Tests for scgrad.layers: SCLinear, SCConv2d, sc_relu, SCReLU.

Layers collapse the elementwise primitive graph (sc_mul per product,
sc_add_tree over the fan-in) into one tensor op. These tests pin the
closed form, the equivalence to the primitives, scale bookkeeping
through chained layers, gradient flow, training-noise behavior, and the
conv reference match.
"""

import pytest
import torch
from torch import Tensor, nn

from scgrad.encoding import SCConfig, SCEncodingError, SCNumber, decode, encode
from scgrad.layers import SCConv2d, SCLinear, SCReLU, sc_relu
from scgrad.ops import sc_add_tree, sc_mul

CFG = SCConfig(encoding="bipolar", length=256, seed=0)


def _set_params(layer: nn.Module, weight: Tensor, bias: Tensor | None = None) -> None:
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            assert layer.bias is not None
            layer.bias.copy_(bias)


class TestSCLinearForward:
    def test_eval_value_matches_closed_form_exactly(self) -> None:
        torch.manual_seed(0)
        layer = SCLinear(4, 2, bias=True, config=CFG).eval()
        w = torch.tensor([[0.5, -0.25, 0.75, -1.0], [0.1, 0.2, -0.3, 0.4]])
        b = torch.tensor([0.25, -0.5])
        _set_params(layer, w, b)
        x = torch.tensor([[0.5, -0.5, 0.25, -0.25], [1.0, -1.0, 0.0, 0.5]])
        out = layer(x)
        assert isinstance(out, SCNumber)
        assert layer.fan_in == 5
        expected = (x @ w.t() + b) / 5
        torch.testing.assert_close(out.value, expected, rtol=0.0, atol=0.0)
        assert out.scale == pytest.approx(1.0 / 5)
        assert out.corr_id == layer.output_corr_id

    def test_decode_recovers_affine_map_for_in_range_params(self) -> None:
        torch.manual_seed(1)
        layer = SCLinear(4, 2, bias=True, config=CFG).eval()
        w = torch.empty(2, 4).uniform_(-1.0, 1.0)
        b = torch.empty(2).uniform_(-1.0, 1.0)
        _set_params(layer, w, b)
        x = torch.empty(3, 4).uniform_(-1.0, 1.0)
        out = layer(x)
        torch.testing.assert_close(decode(out, descale=True), x @ w.t() + b)

    def test_out_of_range_params_are_clamped(self) -> None:
        layer = SCLinear(2, 1, bias=True, config=CFG).eval()
        _set_params(layer, torch.tensor([[1.5, -2.0]]), torch.tensor([3.0]))
        x = torch.tensor([[0.5, 0.5]])
        out = layer(x)
        expected = (x @ torch.tensor([[1.0, -1.0]]).t() + torch.tensor([1.0])) / 3
        torch.testing.assert_close(out.value, expected, rtol=0.0, atol=0.0)

    def test_shapes_batched_and_3d(self) -> None:
        layer = SCLinear(4, 2, bias=True, config=CFG).eval()
        out2d = layer(torch.zeros(3, 4))
        assert out2d.value.shape == (3, 2)
        out3d = layer(torch.zeros(5, 3, 4))
        assert out3d.value.shape == (5, 3, 2)

    def test_no_bias_fan_in_scale_and_value(self) -> None:
        torch.manual_seed(2)
        layer = SCLinear(4, 2, bias=False, config=CFG).eval()
        assert layer.fan_in == 4
        w = torch.empty(2, 4).uniform_(-1.0, 1.0)
        _set_params(layer, w)
        x = torch.empty(3, 4).uniform_(-1.0, 1.0)
        out = layer(x)
        torch.testing.assert_close(out.value, (x @ w.t()) / 4, rtol=0.0, atol=0.0)
        assert out.scale == pytest.approx(1.0 / 4)


class TestSCLinearPrimitiveEquivalence:
    def test_matches_encode_mul_tree_composition(self) -> None:
        torch.manual_seed(3)
        layer = SCLinear(4, 2, bias=True, config=CFG).eval()
        w = torch.empty(2, 4).uniform_(-1.0, 1.0)
        b = torch.empty(2).uniform_(-1.0, 1.0)
        _set_params(layer, w, b)
        x = torch.empty(4).uniform_(-1.0, 1.0)
        out = layer(x)
        assert out.value.shape == (2,)
        for j in range(2):
            terms = [sc_mul(encode(x[i], CFG), encode(w[j, i], CFG)) for i in range(4)]
            terms.append(encode(b[j], CFG))
            tree = sc_add_tree(terms)
            torch.testing.assert_close(tree.value, out.value[j], rtol=1e-6, atol=1e-7)
            assert tree.scale == pytest.approx(out.scale)


class TestSCLinearTraining:
    def test_sgd_step_on_decoded_mse_reduces_loss(self) -> None:
        torch.manual_seed(4)
        layer = SCLinear(4, 2, bias=True, config=CFG, decode_output=True)
        x = torch.empty(8, 4).uniform_(-1.0, 1.0)
        target = torch.empty(8, 2).uniform_(-1.0, 1.0)
        opt = torch.optim.SGD(layer.parameters(), lr=0.05)
        pred = layer(x)
        assert isinstance(pred, Tensor)
        loss0 = nn.functional.mse_loss(pred, target)
        loss0.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0
        assert layer.bias is not None
        assert layer.bias.grad is not None
        opt.step()
        with torch.no_grad():
            loss1 = nn.functional.mse_loss(layer(x), target)
        assert loss1.item() < loss0.item()

    def test_noise_varies_in_train_mode_and_is_off_in_eval(self) -> None:
        cfg = SCConfig(encoding="bipolar", length=64, seed=0, noise=True)
        torch.manual_seed(5)
        layer = SCLinear(4, 2, bias=True, config=cfg)
        w = torch.empty(2, 4).uniform_(-0.8, 0.8)
        b = torch.empty(2).uniform_(-0.8, 0.8)
        _set_params(layer, w, b)
        x = torch.empty(5, 4).uniform_(-0.5, 0.5)
        layer.train()
        torch.manual_seed(6)
        v1 = layer(x).value
        v2 = layer(x).value
        assert not torch.equal(v1, v2)
        layer.eval()
        e1 = layer(x).value
        e2 = layer(x).value
        assert torch.equal(e1, e2)
        expected = (x @ w.t() + b) / 5
        torch.testing.assert_close(e1, expected, rtol=0.0, atol=0.0)


class TestSCLinearChaining:
    def test_scnumber_input_keeps_scale_bookkeeping(self) -> None:
        torch.manual_seed(7)
        layer1 = SCLinear(4, 3, bias=True, config=CFG).eval()
        layer2 = SCLinear(3, 2, bias=True, config=CFG).eval()
        w1 = torch.empty(3, 4).uniform_(-1.0, 1.0)
        b1 = torch.empty(3).uniform_(-1.0, 1.0)
        w2 = torch.empty(2, 3).uniform_(-1.0, 1.0)
        b2 = torch.empty(2).uniform_(-1.0, 1.0)
        _set_params(layer1, w1, b1)
        _set_params(layer2, w2, b2)
        x = torch.empty(2, 4).uniform_(-1.0, 1.0)
        out1 = layer1(x)
        out2 = layer2(out1)
        assert isinstance(out2, SCNumber)
        assert out2.scale == pytest.approx(1.0 / (layer1.fan_in * layer2.fan_in))
        assert out2.corr_id == layer2.output_corr_id
        inner = (x @ w1.t() + b1) / layer1.fan_in
        expected_val = (inner @ w2.t() + out1.scale * b2) / layer2.fan_in
        torch.testing.assert_close(out2.value, expected_val)
        ref = nn.functional.linear(nn.functional.linear(x, w1, b1), w2, b2)
        torch.testing.assert_close(decode(out2, descale=True), ref, rtol=1e-5, atol=1e-5)


class TestSCConv2d:
    def test_value_and_decode_match_reference_conv(self) -> None:
        torch.manual_seed(8)
        layer = SCConv2d(2, 3, kernel_size=3, bias=True, config=CFG).eval()
        w = torch.empty(3, 2, 3, 3).uniform_(-1.0, 1.0)
        b = torch.empty(3).uniform_(-1.0, 1.0)
        _set_params(layer, w, b)
        x = torch.empty(2, 2, 5, 5).uniform_(-1.0, 1.0)
        out = layer(x)
        assert isinstance(out, SCNumber)
        k = layer.fan_in
        assert k == 2 * 3 * 3 + 1
        ref = nn.functional.conv2d(x, w, bias=b)
        assert out.value.shape == ref.shape == (2, 3, 3, 3)
        torch.testing.assert_close(out.value, ref / k)
        assert out.scale == pytest.approx(1.0 / k)
        torch.testing.assert_close(decode(out, descale=True), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        ("stride", "padding"),
        [(1, 0), (2, 1), ((2, 1), (0, 1))],
    )
    def test_stride_padding_variants_match_reference(
        self, stride: int | tuple[int, int], padding: int | tuple[int, int]
    ) -> None:
        torch.manual_seed(9)
        layer = SCConv2d(
            1, 2, kernel_size=(2, 3), stride=stride, padding=padding, bias=True, config=CFG
        ).eval()
        w = torch.empty(2, 1, 2, 3).uniform_(-1.0, 1.0)
        b = torch.empty(2).uniform_(-1.0, 1.0)
        _set_params(layer, w, b)
        x = torch.empty(2, 1, 6, 7).uniform_(-1.0, 1.0)
        out = layer(x)
        ref = nn.functional.conv2d(x, w, bias=b, stride=stride, padding=padding)
        k = layer.fan_in
        assert k == 1 * 2 * 3 + 1
        assert out.value.shape == ref.shape
        torch.testing.assert_close(out.value, ref / k)
        assert out.scale == pytest.approx(1.0 / k)

    def test_groups_not_one_raises(self) -> None:
        with pytest.raises(SCEncodingError):
            SCConv2d(2, 2, kernel_size=1, groups=2, config=CFG)


class TestSCReLU:
    def test_sc_relu_applies_relu_to_value_and_preserves_scale(self) -> None:
        val = torch.tensor([-0.5, 0.25, -0.1, 0.8])
        s = SCNumber(val, CFG, scale=0.2)
        out = sc_relu(s)
        torch.testing.assert_close(out.value, torch.relu(val), rtol=0.0, atol=0.0)
        assert out.scale == 0.2
        assert out.corr_id == s.corr_id

    def test_screlu_module_wraps_sc_relu(self) -> None:
        val = torch.tensor([[-1.0, 0.5], [0.0, -0.25]])
        out = SCReLU()(SCNumber(val, CFG, scale=0.5))
        torch.testing.assert_close(out.value, torch.relu(val), rtol=0.0, atol=0.0)
        assert out.scale == 0.5
