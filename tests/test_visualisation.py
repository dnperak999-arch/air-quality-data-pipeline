import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.analysis import (
    POLLUTANT_COLUMNS,
    fit_standardised_pca,
    spearman_correlation,
    standardise_pollutants,
)
from src.visualisation import (
    _extreme_particulate_index,
    plot_pca_snapshot,
    plot_pm_scatter,
    plot_pollutant_heatmap,
    plot_spearman_heatmap,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_snapshot_frame(n_rows: int = 8) -> pd.DataFrame:
    values = np.random.default_rng(1).uniform(
        0.2,
        40,
        size=(n_rows, len(POLLUTANT_COLUMNS)),
    )
    frame = pd.DataFrame(values, columns=POLLUTANT_COLUMNS)
    frame["city"] = [f"City {number}" for number in range(n_rows)]
    frame["aqi"] = (np.arange(n_rows) % 3) + 1
    return frame


def assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(PNG_SIGNATURE)
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "plot",
    [
        plot_pollutant_heatmap,
        plot_pm_scatter,
        plot_pca_snapshot,
        plot_spearman_heatmap,
    ],
)
def test_plot_writes_png_and_creates_parent_directories(tmp_path, plot):
    frame = make_snapshot_frame()
    original = frame.copy(deep=True)
    output = tmp_path / "nested" / "figures" / "snapshot.png"

    plot(frame, output)

    assert output.parent.is_dir()
    assert_png(output)
    pd.testing.assert_frame_equal(frame, original)


def test_heatmap_uses_analysis_standardisation_without_changing_input(tmp_path, monkeypatch):
    frame = make_snapshot_frame()
    original = frame.copy(deep=True)
    calls = {"count": 0}

    def spy(df):
        calls["count"] += 1
        pd.testing.assert_frame_equal(df, original)
        return standardise_pollutants(df)

    monkeypatch.setattr("src.visualisation.standardise_pollutants", spy)

    plot_pollutant_heatmap(frame, tmp_path / "heatmap.png")

    assert calls["count"] == 1
    pd.testing.assert_frame_equal(frame, original)
    assert plt.get_fignums() == []


def test_pm_scatter_uses_original_concentrations(tmp_path, monkeypatch):
    frame = make_snapshot_frame()
    frame["pm2_5"] = [1.0, 2.0, 3.0, 8.0, 5.0, 4.0, 6.0, 1.5]
    frame["pm10"] = [2.0, 2.5, 4.0, 9.0, 7.0, 3.0, 8.0, 2.2]
    original = frame.copy(deep=True)

    def fail_if_scaled(_df):
        raise AssertionError("PM scatter must not standardise concentrations")

    monkeypatch.setattr("src.visualisation.standardise_pollutants", fail_if_scaled)

    plot_pm_scatter(frame, tmp_path / "pm.png")

    assert_png(tmp_path / "pm.png")
    pd.testing.assert_frame_equal(frame, original)


def test_extreme_particulate_index_is_deterministic():
    pm2_5 = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0], index=list("abcde"))
    pm10 = pd.Series([1.0, 2.0, 3.0, 80.0, 4.0], index=list("abcde"))

    assert _extreme_particulate_index(pm2_5, pm10) == ["a", "e", "d"]


def test_pca_plot_fits_once_and_leaves_input_unchanged(tmp_path, monkeypatch):
    frame = make_snapshot_frame()
    original = frame.copy(deep=True)
    calls = {"count": 0}

    def spy(df, n_components=2):
        calls["count"] += 1
        calls["n_components"] = n_components
        return fit_standardised_pca(df, n_components=n_components)

    monkeypatch.setattr("src.visualisation.fit_standardised_pca", spy)

    plot_pca_snapshot(frame, tmp_path / "pca.png")

    assert calls == {"count": 1, "n_components": 2}
    assert_png(tmp_path / "pca.png")
    pd.testing.assert_frame_equal(frame, original)


def test_spearman_plot_uses_analysis_correlation(tmp_path, monkeypatch):
    frame = make_snapshot_frame()
    original = frame.copy(deep=True)
    calls = {"count": 0}

    def spy(df):
        calls["count"] += 1
        return spearman_correlation(df)

    monkeypatch.setattr("src.visualisation.spearman_correlation", spy)

    plot_spearman_heatmap(frame, tmp_path / "spearman.png")

    assert calls["count"] == 1
    assert_png(tmp_path / "spearman.png")
    pd.testing.assert_frame_equal(frame, original)


def test_repeated_plots_do_not_leave_figures_open(tmp_path):
    frame = make_snapshot_frame()

    plot_pollutant_heatmap(frame, tmp_path / "a.png")
    plot_pm_scatter(frame, tmp_path / "b.png")
    plot_pca_snapshot(frame, tmp_path / "c.png")
    plot_spearman_heatmap(frame, tmp_path / "d.png")

    assert plt.get_fignums() == []
