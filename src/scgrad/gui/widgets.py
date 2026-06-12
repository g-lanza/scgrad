"""Widgets for the scgrad instrument: every panel shows a real quantity.

Palette: phosphor on slate. Teal means the math is behaving (independent,
converged, within tolerance); burnt amber means correlated or diverging;
oxide red is reserved for hard failure. Color encodes facts, nothing else.
"""

from __future__ import annotations

import math

from rich.text import Text
from textual.widgets import Sparkline, Static

TEAL = "#4DB6A0"
AMBER = "#D98A3D"
RED = "#C7503F"
TEXT = "#C8D2DC"
DIM = "#6B7884"

_ON = "▆"
_OFF = "·"


class BitstreamRaster(Static):
    """A value's actual bit pattern as a row of lit/unlit cells, streaming.

    The signature element: each row is one SC value; lit cells are ones.
    The window slides along the real stream every refresh, so the
    probability flickers past as bits, which is what stochastic
    computing is.
    """

    def update_streams(self, rows: list[tuple[str, list[int], float]]) -> None:
        """rows: (label, bit window, p_hat estimate) per displayed value."""
        text = Text()
        for label, bits, p_hat in rows:
            text.append(f"{label:>10} ", style=DIM)
            for b in bits:
                text.append(_ON if b else _OFF, style=TEAL if b else DIM)
            text.append(f"  p̂={p_hat:+.3f}\n", style=TEXT)
        self.update(text)


class CorrelationHeatmap(Static):
    """Pairwise SCC matrix of the live port streams, teal to amber.

    SCC near 0 (independent) renders teal; |SCC| near 1 (correlated)
    renders amber. The diagonal is by definition +1 and shown dim.
    """

    def update_matrix(self, labels: list[str], matrix: list[list[float]]) -> None:
        text = Text()
        text.append(" " * 7, style=DIM)
        for label in labels:
            text.append(f"{label:>7}", style=DIM)
        text.append("\n")
        for i, row_label in enumerate(labels):
            text.append(f"{row_label:>7}", style=DIM)
            for j, value in enumerate(matrix[i]):
                if i == j:
                    text.append(f"{value:+7.2f}", style=DIM)
                else:
                    style = AMBER if abs(value) > 0.2 else TEAL
                    text.append(f"{value:+7.2f}", style=style)
            text.append("\n")
        self.update(text)


class AccuracyCurve(Static):
    """eps(L): the analytic error bound against measured points, log-log."""

    def update_curve(
        self,
        analytic: list[tuple[int, float]],
        measured: list[tuple[int, float]],
        k_terms: int,
    ) -> None:
        try:
            import plotext
        except ImportError:
            self.update(Text("plotext missing: install scgrad[gui]", style=RED))
            return
        plotext.clear_figure()
        plotext.theme("pro")
        plotext.plot_size(max(self.size.width - 4, 30), max(self.size.height - 4, 8))
        plotext.plot(
            [math.log2(n) for n, _ in analytic],
            [math.log10(e) for _, e in analytic],
            marker="braille",
            color=(107, 120, 132),
            label="analytic",
        )
        if measured:
            plotext.scatter(
                [math.log2(n) for n, _ in measured],
                [math.log10(max(e, 1e-9)) for _, e in measured],
                marker="x",
                color=(77, 182, 160),
                label="measured",
            )
        plotext.xlabel("log2 L")
        plotext.ylabel("log10 eps")
        body = plotext.build()
        text = Text.from_ansi(body)
        text.append(
            f"\n eps_max = (k-1)/(L*k), k={k_terms} (uGEMM scaled-add bound)",
            style=DIM,
        )
        self.update(text)


class LossPanel(Static):
    """Task loss, correlation penalty, and their sum, with history."""

    def compose_text(
        self, task: float, corr: float, lam: float, n_rngs: int | None, step: int
    ) -> Text:
        total = task + lam * corr
        text = Text()
        text.append(f" step {step:>5}  ", style=DIM)
        text.append(f"task {task:8.5f}", style=TEXT)
        text.append("  +  ")
        corr_style = AMBER if corr > 1e-6 else TEAL
        text.append(f"{lam:g}·corr {corr:8.5f}", style=corr_style)
        text.append("  =  ")
        text.append(f"{total:8.5f}", style=TEXT)
        budget = "unbounded" if n_rngs is None else str(n_rngs)
        text.append(f"   rng budget: {budget}", style=DIM)
        return text


class LossSparkline(Sparkline):
    """Loss history sparkline (data encodes the trend; no decoration)."""
