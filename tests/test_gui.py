"""Headless smoke test: the instrument constructs and renders real data."""

import pytest

pytest.importorskip("textual")
pytest.importorskip("plotext")


def test_app_renders_real_data() -> None:
    import asyncio

    from scgrad.gui.app import ScgradApp
    from scgrad.gui.widgets import BitstreamRaster, CorrelationHeatmap

    async def run() -> None:
        app = ScgradApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(1.5)
            assert app.run_state.step >= 1
            assert app.run_state.task_loss == app.run_state.task_loss  # not NaN
            raster = app.query_one("#raster", BitstreamRaster)
            assert len(str(raster.render())) > 0
            heatmap = app.query_one("#heatmap", CorrelationHeatmap)
            rendered = str(heatmap.render())
            assert "L1.w" in rendered
            assert app.run_state.corr_penalty > 0.0

    asyncio.run(run())
