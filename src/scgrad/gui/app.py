"""The scgrad instrument: a dense terminal panel over a live SC training run.

Boots straight into the instrument: a real two-layer SC model training on
a fixed toy regression task, with a rng budget of 2 so one multiply is
genuinely correlated and the correlation penalty has something to do.
Every panel shows a live quantity from that run: the actual bitstreams of
the first-layer output, the measured SCC matrix of the port streams, the
analytic eps(L) bound against the measured dual-path error, and the loss
decomposition. No onboarding, no hero, no decoration.
"""

from __future__ import annotations

import torch
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal
from textual.widgets import Static

from scgrad.accuracy import accuracy_estimator
from scgrad.correlation import CorrelationTracker, correlation_loss, scc
from scgrad.encoding import SCConfig, SCNumber, decode
from scgrad.eval_exact import exact_forward
from scgrad.gui.widgets import (
    DIM,
    TEXT,
    AccuracyCurve,
    BitstreamRaster,
    CorrelationHeatmap,
    LossPanel,
    LossSparkline,
)
from scgrad.hardware import make_source
from scgrad.layers import SCLinear, SCReLU

_LAMBDA = 0.5
_WINDOW = 56


class _Run:
    """The live training run the instrument observes."""

    def __init__(self) -> None:
        torch.manual_seed(7)
        self.config = SCConfig(encoding="bipolar", length=256, seed=7, n_rngs=2, noise=True)
        self.model = torch.nn.Sequential(
            SCLinear(12, 6, config=self.config),
            SCReLU(),
            SCLinear(6, 1, config=self.config),
        )
        self.x = torch.rand(64, 12) * 2.0 - 1.0
        target = (self.x[:, :6].sum(dim=1) - self.x[:, 6:].sum(dim=1)) / 12.0
        self.y = target.unsqueeze(1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        self.step = 0
        self.task_loss = float("nan")
        self.corr_penalty = float("nan")
        self.history: list[float] = []
        self.offset = 0

    def train_step(self) -> None:
        self.model.train()
        with CorrelationTracker() as tracker:
            out = self.model(self.x)
        assert isinstance(out, SCNumber)
        task = torch.nn.functional.mse_loss(decode(out), self.y)
        corr = correlation_loss(tracker)
        loss = task + _LAMBDA * corr
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        self.step += 1
        self.task_loss = float(task.item())
        self.corr_penalty = float(corr.item())
        self.history.append(float(loss.item()))
        self.offset = (self.offset + 8) % (self.config.length - _WINDOW)

    def first_layer(self) -> SCLinear:
        layer = self.model[0]
        assert isinstance(layer, SCLinear)
        return layer

    def raster_rows(self) -> list[tuple[str, list[int], float]]:
        """Real bit windows of the first layer's output neurons."""
        self.model.eval()
        with torch.no_grad():
            out = self.model[0](self.x[:1])
        assert isinstance(out, SCNumber)
        source = make_source(self.config)
        bits = source.bits(out.probabilities()[0], self.config.length, out.corr_id)
        rows = []
        for i in range(min(4, bits.shape[0])):
            window = bits[i, self.offset : self.offset + _WINDOW].tolist()
            p_hat = float(bits[i].float().mean().item())
            rows.append((f"h{i} bits", [int(b) for b in window], 2.0 * p_hat - 1.0))
        return rows

    def port_scc(self) -> tuple[list[str], list[list[float]]]:
        """Measured SCC between the model's port streams under the rng budget."""
        layer1 = self.first_layer()
        layer2 = self.model[2]
        assert isinstance(layer2, SCLinear)
        source = make_source(self.config)
        n = self.config.length
        p = torch.full((1,), 0.5)
        ports = [
            ("L1.x", layer1.input_corr_id),
            ("L1.w", layer1.weight_corr_id),
            ("L2.x", layer1.output_corr_id),
            ("L2.w", layer2.weight_corr_id),
        ]
        streams = [source.bits(p, n, cid)[0] for _, cid in ports]
        labels = [name for name, _ in ports]
        matrix = [
            [float(scc(streams[i], streams[j]).item()) for j in range(len(ports))]
            for i in range(len(ports))
        ]
        return labels, matrix

    def dual_path_point(self) -> tuple[int, float]:
        """Measured mean |approx - exact| of the model at the current length."""
        self.model.eval()
        with torch.no_grad():
            approx = self.model(self.x[:8])
        assert isinstance(approx, SCNumber)
        exact = exact_forward(self.model, self.x[:8], self.config)
        err = float((approx.value - exact.value).abs().mean().item())
        return self.config.length, err


class ScgradApp(App[None]):
    """The instrument."""

    CSS = f"""
    Screen {{
        background: #0A0E12;
        color: {TEXT};
    }}
    #status {{
        height: 1;
        background: #11161C;
        color: {DIM};
        padding: 0 1;
    }}
    Grid {{
        grid-size: 2 2;
        grid-gutter: 0;
    }}
    .panel {{
        background: #11161C;
        border: solid #1F2730;
        padding: 0 1;
    }}
    .panel-title {{
        color: {DIM};
        height: 1;
    }}
    #loss-row {{
        height: 4;
        background: #11161C;
        border: solid #1F2730;
    }}
    #loss-spark {{
        width: 1fr;
        margin: 0 1;
    }}
    Sparkline > .sparkline--max-color {{
        color: #4DB6A0;
    }}
    Sparkline > .sparkline--min-color {{
        color: #1F2730;
    }}
    """

    TITLE = "scgrad"

    def __init__(self) -> None:
        super().__init__()
        self.run_state = _Run()

    def compose(self) -> ComposeResult:
        cfg = self.run_state.config
        yield Static(
            f"scgrad · {cfg.encoding} · L={cfg.length} · source={cfg.source} · "
            f"rngs={cfg.n_rngs} · noise={'on' if cfg.noise else 'off'} · depth 2",
            id="status",
        )
        with Grid():
            with Static(classes="panel") as p1:
                p1.border_title = "BITSTREAM INSPECTOR"
                yield BitstreamRaster(id="raster")
            with Static(classes="panel") as p2:
                p2.border_title = "CORRELATION MATRIX (SCC)"
                yield CorrelationHeatmap(id="heatmap")
            with Static(classes="panel") as p3:
                p3.border_title = "NETWORK"
                yield Static(id="network")
            with Static(classes="panel") as p4:
                p4.border_title = "ACCURACY ESTIMATOR"
                yield AccuracyCurve(id="curve")
        with Horizontal(id="loss-row"):
            yield LossPanel(id="loss")
            yield LossSparkline(id="loss-spark", summary_function=min)

    def on_mount(self) -> None:
        self._refresh_static()
        self.set_interval(0.5, self._tick)

    def _refresh_static(self) -> None:
        run = self.run_state
        layer1 = run.first_layer()
        layer2 = run.model[2]
        assert isinstance(layer2, SCLinear)
        network = (
            f"SCLinear {layer1.in_features}→{layer1.out_features}   scale 1/{layer1.fan_in}\n"
            f"SCReLU\n"
            f"SCLinear {layer2.in_features}→{layer2.out_features}   scale 1/{layer2.fan_in}\n"
            f"Σ scale 1/{layer1.fan_in * layer2.fan_in}"
        )
        self.query_one("#network", Static).update(network)

    def _tick(self) -> None:
        run = self.run_state
        run.train_step()
        self.query_one("#raster", BitstreamRaster).update_streams(run.raster_rows())
        labels, matrix = run.port_scc()
        self.query_one("#heatmap", CorrelationHeatmap).update_matrix(labels, matrix)
        estimator = accuracy_estimator(2, [f"add:{run.first_layer().fan_in}"], "bipolar")
        curve = estimator.curve([64, 128, 256, 512, 1024, 2048, 4096])
        measured = [run.dual_path_point()] if run.step % 8 == 1 else []
        if measured:
            self._last_measured = measured
        shown = getattr(self, "_last_measured", [])
        self.query_one("#curve", AccuracyCurve).update_curve(curve, shown, run.first_layer().fan_in)
        loss_panel = self.query_one("#loss", LossPanel)
        loss_panel.update(
            loss_panel.compose_text(
                run.task_loss, run.corr_penalty, _LAMBDA, run.config.n_rngs, run.step
            )
        )
        spark = self.query_one("#loss-spark", LossSparkline)
        spark.data = run.history[-120:]


def main() -> None:
    """Console entry point: scgrad-gui."""
    ScgradApp().run()


if __name__ == "__main__":
    main()
