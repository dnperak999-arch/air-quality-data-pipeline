from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


POLLUTANT_COLUMNS = [
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
]


@dataclass
class PCAResult:
    """Oriented PCA of the eight standardised pollutant columns.

    ``scores`` is indexed like the input rows. ``loadings`` is indexed by
    pollutant. Both use component names ``PC1``, ``PC2``, and so on.
    Variance ratios are unaffected by the sign convention.
    """

    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance_ratio: pd.Series
    cumulative_explained_variance: pd.Series


def summarise_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Return min, median, mean, max, population std, and skewness.

    ``std`` uses the population standard deviation (divisor n), matching
    ``StandardScaler``. ``skewness`` is pandas' adjusted Fisher-Pearson
    coefficient from ``DataFrame.skew``.
    """

    pollutants = _validate_pollutant_frame(df)
    summary = pd.DataFrame(
        {
            "min": pollutants.min(),
            "median": pollutants.median(),
            "mean": pollutants.mean(),
            "max": pollutants.max(),
            "std": pollutants.std(ddof=0),
            "skewness": pollutants.skew(),
        }
    )
    summary.index.name = "pollutant"
    return summary


def standardise_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new z-scored frame for the eight pollutant columns.

    Scaling uses ``StandardScaler`` (center by the mean, divide by the
    population standard deviation). The input frame is not modified and
    nothing is written to disk.
    """

    pollutants = _validate_pollutant_frame(df)
    scaled = StandardScaler().fit_transform(pollutants.to_numpy(dtype=float))
    return pd.DataFrame(
        scaled,
        index=pollutants.index,
        columns=POLLUTANT_COLUMNS,
    )


def spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 8x8 Spearman correlation of the pollutant columns."""

    pollutants = _validate_pollutant_frame(df)
    correlation = pollutants.corr(method="spearman")
    return correlation.loc[POLLUTANT_COLUMNS, POLLUTANT_COLUMNS]


def fit_standardised_pca(df: pd.DataFrame, n_components: int = 2) -> PCAResult:
    """Fit PCA on internally standardised pollutants and orient the signs.

    ``n_components`` must be an integer from 1 through
    ``min(n_rows, 8)``. The same input always yields the same oriented
    result. The input frame is not modified.
    """

    pollutants = _validate_pollutant_frame(df)
    n_components = _validate_n_components(n_components, n_rows=len(pollutants))
    component_names = [f"PC{number}" for number in range(1, n_components + 1)]

    scaled = StandardScaler().fit_transform(pollutants.to_numpy(dtype=float))
    pca = PCA(n_components=n_components, svd_solver="full")
    score_values = pca.fit_transform(scaled)

    scores = pd.DataFrame(
        score_values,
        index=pollutants.index,
        columns=component_names,
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=pd.Index(POLLUTANT_COLUMNS, name="pollutant"),
        columns=component_names,
    )
    scores, loadings = orient_components(scores, loadings)

    explained_variance_ratio = pd.Series(
        pca.explained_variance_ratio_,
        index=pd.Index(component_names, name="component"),
        name="explained_variance_ratio",
    )
    cumulative_explained_variance = pd.Series(
        np.cumsum(pca.explained_variance_ratio_),
        index=explained_variance_ratio.index,
        name="cumulative_explained_variance",
    )
    return PCAResult(
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance=cumulative_explained_variance,
    )


def orient_components(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make each component's largest-magnitude loading positive.

    When a loading column is flipped, the matching score column is flipped
    too. Sign flips do not change explained variance.
    """

    oriented_scores = scores.copy()
    oriented_loadings = loadings.copy()

    for component in oriented_loadings.columns:
        loading_values = oriented_loadings[component].to_numpy(dtype=float)
        anchor = int(np.argmax(np.abs(loading_values)))
        if loading_values[anchor] < 0:
            oriented_loadings[component] = -oriented_loadings[component]
            oriented_scores[component] = -oriented_scores[component]

    return oriented_scores, oriented_loadings


def _validate_pollutant_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the eight pollutant columns, or raise ValueError."""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Analysis input must be a pandas DataFrame.")

    missing = [column for column in POLLUTANT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing pollutant columns: {missing}")

    if len(df.index) < 1:
        raise ValueError("Analysis input must contain at least one row.")

    pollutants = df.loc[:, POLLUTANT_COLUMNS].copy()
    for column in POLLUTANT_COLUMNS:
        dtype_is_numeric = pd.api.types.is_numeric_dtype(pollutants[column])
        dtype_is_boolean = pd.api.types.is_bool_dtype(pollutants[column])
        if dtype_is_boolean or not dtype_is_numeric:
            raise ValueError(f"Pollutant column {column!r} must be numeric.")

    if pollutants.isna().any().any():
        raise ValueError("Pollutant values contain NaN.")

    if not np.isfinite(pollutants.to_numpy(dtype=float)).all():
        raise ValueError("Pollutant values contain infinity.")

    return pollutants


def _validate_n_components(n_components: int, n_rows: int) -> int:
    """Return n_components when it is in range, or raise ValueError."""

    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise ValueError(f"n_components must be an integer, got {n_components!r}.")

    maximum = min(n_rows, len(POLLUTANT_COLUMNS))
    if n_components < 1 or n_components > maximum:
        raise ValueError(
            f"n_components must be between 1 and {maximum}, got {n_components}."
        )
    return int(n_components)
