import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.analysis import (
    POLLUTANT_COLUMNS,
    fit_standardised_pca,
    orient_components,
    spearman_correlation,
    standardise_pollutants,
    summarise_pollutants,
)


CANONICAL_DATASET = PROJECT_ROOT / "data" / "processed" / "air_quality_data.csv"
SUMMARY_COLUMNS = ["min", "median", "mean", "max", "std", "skewness"]


def make_pollutant_frame(n_rows=12, index=None) -> pd.DataFrame:
    values = np.random.default_rng(0).normal(
        loc=np.arange(len(POLLUTANT_COLUMNS)),
        scale=1.5,
        size=(n_rows, len(POLLUTANT_COLUMNS)),
    )
    frame = pd.DataFrame(values, columns=POLLUTANT_COLUMNS, index=index)
    frame["aqi"] = (np.arange(n_rows) % 5) + 1
    frame["city"] = [f"city-{number}" for number in range(n_rows)]
    return frame


def fit_two_components(frame: pd.DataFrame):
    return fit_standardised_pca(frame, n_components=2)


def analysis_calls():
    return [
        summarise_pollutants,
        standardise_pollutants,
        spearman_correlation,
        fit_two_components,
    ]


def test_pollutant_columns_are_the_eight_canonical_names():
    assert POLLUTANT_COLUMNS == [
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
    ]


@pytest.mark.parametrize("analyse", analysis_calls())
def test_analysis_functions_do_not_mutate_input(analyse):
    frame = make_pollutant_frame()
    original = frame.copy(deep=True)

    analyse(frame)

    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("analyse", analysis_calls())
def test_missing_pollutant_column_raises(analyse):
    frame = make_pollutant_frame().drop(columns=["nh3"])

    with pytest.raises(ValueError, match="Missing pollutant columns"):
        analyse(frame)


@pytest.mark.parametrize("analyse", analysis_calls())
def test_nan_pollutant_value_raises(analyse):
    frame = make_pollutant_frame()
    frame.loc[frame.index[0], "pm2_5"] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        analyse(frame)


@pytest.mark.parametrize("infinite_value", [np.inf, -np.inf])
@pytest.mark.parametrize("analyse", analysis_calls())
def test_infinite_pollutant_value_raises(analyse, infinite_value):
    frame = make_pollutant_frame()
    frame.loc[frame.index[1], "so2"] = infinite_value

    with pytest.raises(ValueError, match="infinity"):
        analyse(frame)


def test_summarise_pollutants_uses_population_standard_deviation():
    frame = make_pollutant_frame(n_rows=3)
    frame["co"] = [1.0, 2.0, 3.0]

    summary = summarise_pollutants(frame)

    assert list(summary.index) == POLLUTANT_COLUMNS
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert "aqi" not in summary.columns
    assert summary.loc["co", "min"] == pytest.approx(1.0)
    assert summary.loc["co", "median"] == pytest.approx(2.0)
    assert summary.loc["co", "mean"] == pytest.approx(2.0)
    assert summary.loc["co", "max"] == pytest.approx(3.0)
    assert summary.loc["co", "std"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=0))
    assert summary.loc["co", "skewness"] == pytest.approx(0.0)


def test_standardise_pollutants_returns_z_scores_without_metadata():
    frame = make_pollutant_frame(
        index=pd.Index(range(10, 22), name="observation"),
    )

    scaled = standardise_pollutants(frame)

    assert len(scaled) == len(frame)
    assert list(scaled.columns) == POLLUTANT_COLUMNS
    assert scaled.index.equals(frame.index)
    assert np.allclose(scaled.mean().to_numpy(), 0.0, atol=1e-8)
    assert np.allclose(scaled.std(ddof=0).to_numpy(), 1.0, atol=1e-8)
    assert np.isfinite(scaled.to_numpy()).all()


def test_standardise_pollutants_keeps_constant_column_finite():
    frame = make_pollutant_frame()
    frame["no"] = 0.0

    scaled = standardise_pollutants(frame)
    varying = [column for column in POLLUTANT_COLUMNS if column != "no"]

    assert np.isfinite(scaled.to_numpy()).all()
    assert np.allclose(scaled["no"].to_numpy(), 0.0, atol=1e-8)
    assert np.allclose(scaled[varying].std(ddof=0).to_numpy(), 1.0, atol=1e-8)


def test_spearman_correlation_is_labelled_and_bounded():
    correlation = spearman_correlation(make_pollutant_frame())

    assert correlation.shape == (8, 8)
    assert list(correlation.index) == POLLUTANT_COLUMNS
    assert list(correlation.columns) == POLLUTANT_COLUMNS
    assert "aqi" not in correlation.columns
    assert np.allclose(correlation.to_numpy(), correlation.to_numpy().T)
    assert np.allclose(np.diag(correlation.to_numpy()), 1.0)
    assert correlation.to_numpy().min() >= -1.0
    assert correlation.to_numpy().max() <= 1.0


def test_pca_result_has_expected_shape_labels_and_variance():
    frame = make_pollutant_frame()

    result = fit_standardised_pca(frame, n_components=2)
    full = fit_standardised_pca(frame, n_components=8)

    assert result.scores.shape == (len(frame), 2)
    assert result.loadings.shape == (8, 2)
    assert list(result.scores.columns) == ["PC1", "PC2"]
    assert list(result.loadings.columns) == ["PC1", "PC2"]
    assert list(result.loadings.index) == POLLUTANT_COLUMNS
    assert list(result.explained_variance_ratio.index) == ["PC1", "PC2"]
    assert list(result.cumulative_explained_variance.index) == ["PC1", "PC2"]
    assert result.scores.index.equals(frame.index)

    numeric_parts = [
        result.scores.to_numpy(),
        result.loadings.to_numpy(),
        result.explained_variance_ratio.to_numpy(),
        result.cumulative_explained_variance.to_numpy(),
    ]
    assert all(np.isfinite(values).all() for values in numeric_parts)
    assert (result.explained_variance_ratio.to_numpy() > 0).all()
    assert np.all(np.diff(result.cumulative_explained_variance.to_numpy()) >= -1e-12)
    assert full.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert full.cumulative_explained_variance.iloc[-1] == pytest.approx(1.0)

    reconstructed = np.dot(full.scores.to_numpy(), full.loadings.to_numpy().T)
    assert np.allclose(reconstructed, standardise_pollutants(frame).to_numpy())


def test_pca_is_deterministic_after_orientation():
    frame = make_pollutant_frame()

    first = fit_standardised_pca(frame, n_components=3)
    second = fit_standardised_pca(frame, n_components=3)

    pd.testing.assert_frame_equal(first.scores, second.scores)
    pd.testing.assert_frame_equal(first.loadings, second.loadings)
    pd.testing.assert_series_equal(
        first.explained_variance_ratio,
        second.explained_variance_ratio,
    )
    pd.testing.assert_series_equal(
        first.cumulative_explained_variance,
        second.cumulative_explained_variance,
    )


def test_orient_components_flips_scores_with_loadings():
    loadings = pd.DataFrame(
        {
            "PC1": [-0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3],
            "PC2": [0.1, 0.1, 0.1, -0.9, 0.2, 0.0, 0.0, 0.0],
            "PC3": [0.7, -0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1],
        },
        index=POLLUTANT_COLUMNS,
    )
    scores = pd.DataFrame(
        {
            "PC1": [1.5, -0.5],
            "PC2": [2.0, 3.0],
            "PC3": [-1.0, 4.0],
        },
        index=["r1", "r2"],
    )
    original_scores = scores.copy(deep=True)
    original_loadings = loadings.copy(deep=True)

    oriented_scores, oriented_loadings = orient_components(scores, loadings)

    for component in oriented_loadings.columns:
        column = oriented_loadings[component]
        assert column.loc[column.abs().idxmax()] > 0

    assert oriented_loadings.loc["co", "PC1"] == pytest.approx(0.8)
    assert oriented_scores["PC1"].tolist() == pytest.approx([-1.5, 0.5])
    assert oriented_loadings.loc["o3", "PC2"] == pytest.approx(0.9)
    assert oriented_scores["PC2"].tolist() == pytest.approx([-2.0, -3.0])
    assert oriented_loadings.loc["co", "PC3"] == pytest.approx(0.7)
    assert oriented_scores["PC3"].tolist() == pytest.approx([-1.0, 4.0])
    assert np.allclose(
        scores.std(ddof=0).to_numpy(),
        oriented_scores.std(ddof=0).to_numpy(),
    )
    assert np.allclose(
        (loadings.to_numpy() ** 2).sum(axis=0),
        (oriented_loadings.to_numpy() ** 2).sum(axis=0),
    )
    pd.testing.assert_frame_equal(scores, original_scores)
    pd.testing.assert_frame_equal(loadings, original_loadings)


def test_fitted_pca_uses_positive_anchor_loadings():
    result = fit_standardised_pca(make_pollutant_frame(), n_components=4)

    for component in result.loadings.columns:
        column = result.loadings[component]
        assert column.loc[column.abs().idxmax()] > 0


@pytest.mark.parametrize("n_components", [0, -1, 9, 2.5, "2", True, None])
def test_invalid_n_components_raises(n_components):
    frame = make_pollutant_frame()

    with pytest.raises(ValueError, match="n_components"):
        fit_standardised_pca(frame, n_components=n_components)


def test_n_components_cannot_exceed_row_count():
    frame = make_pollutant_frame(n_rows=4)

    with pytest.raises(ValueError, match="between 1 and 4"):
        fit_standardised_pca(frame, n_components=5)

    result = fit_standardised_pca(frame, n_components=4)
    assert result.scores.shape == (4, 4)
    assert result.explained_variance_ratio.sum() == pytest.approx(1.0)


def test_canonical_dataset_passes_analysis_without_rewriting_file():
    digest_before = hashlib.sha256(CANONICAL_DATASET.read_bytes()).hexdigest()
    frame = pd.read_csv(CANONICAL_DATASET)
    original = frame.copy(deep=True)

    summary = summarise_pollutants(frame)
    correlation = spearman_correlation(frame)
    result = fit_standardised_pca(frame, n_components=2)

    digest_after = hashlib.sha256(CANONICAL_DATASET.read_bytes()).hexdigest()
    pd.testing.assert_frame_equal(frame, original)
    assert digest_before == digest_after
    assert summary.shape == (8, len(SUMMARY_COLUMNS))
    assert list(summary.index) == POLLUTANT_COLUMNS
    assert correlation.shape == (8, 8)
    assert result.scores.shape == (len(frame), 2)
    assert result.loadings.shape == (8, 2)
    assert np.isfinite(result.explained_variance_ratio.to_numpy()).all()
    assert np.isfinite(result.cumulative_explained_variance.to_numpy()).all()
