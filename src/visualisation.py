"""Figures for one cross-sectional pollutant snapshot.

Numeric work stays in ``src.analysis``. These functions draw and save.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import (
    POLLUTANT_COLUMNS,
    fit_standardised_pca,
    spearman_correlation,
    standardise_pollutants,
)


POLLUTANT_DISPLAY_NAMES = {
    "co": "CO",
    "no": "NO",
    "no2": "NO2",
    "o3": "O3",
    "so2": "SO2",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nh3": "NH3",
}


def plot_pollutant_heatmap(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a city-by-pollutant heatmap of within-column z-scores.

    Rows are sorted alphabetically by city for reading. The z-scores come
    from ``standardise_pollutants`` and describe this snapshot only.
    """

    _require_columns(df, ["city"])
    scaled = standardise_pollutants(df)
    order = df["city"].sort_values(kind="mergesort").index
    scaled = scaled.loc[order]
    cities = df.loc[order, "city"].astype(str)

    values = scaled.to_numpy(dtype=float)
    limit = float(np.max(np.abs(values))) if values.size else 1.0
    if limit == 0:
        limit = 1.0

    height = max(6.0, 0.36 * len(scaled) + 2.0)
    fig, ax = plt.subplots(figsize=(10.5, height), constrained_layout=True)
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    ax.set_xticks(range(len(POLLUTANT_COLUMNS)))
    ax.set_xticklabels(
        [POLLUTANT_DISPLAY_NAMES[column] for column in POLLUTANT_COLUMNS],
        rotation=0,
    )
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities.tolist())
    ax.set_xlabel("Pollutant")
    ax.set_ylabel("City")
    ax.set_title(
        "Relative pollutant profiles in this city snapshot\n"
        "Colour is a within-pollutant z-score, not absolute severity"
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Standardised value (z-score)")
    _save_figure(fig, output_path)


def plot_pm_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save PM2.5 against PM10 using original concentrations.

    Annotated cities are extremes of either particulate column: the minimum,
    the maximum, or any value at or above that column's 90th percentile.
    Ties are included. A city is labeled once, in input order. Cities that
    share a coordinate are stacked so the labels stay apart. No line is fitted.
    """

    _require_columns(df, ["city", "pm2_5", "pm10"])
    pm2_5 = df["pm2_5"]
    pm10 = df["pm10"]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    ax.scatter(pm2_5, pm10, color="#4C72B0", s=36, zorder=3)
    _annotate_particulate_extremes(ax, df)
    ax.set_xlabel("PM2.5 (original API concentration)")
    ax.set_ylabel("PM10 (original API concentration)")
    ax.set_title(
        "PM2.5 and PM10 in this snapshot\n"
        "Original concentrations; labels mark particulate extremes"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.margins(0.12)
    _save_figure(fig, output_path)


def plot_pca_snapshot(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save PC1/PC2 scores and the pollutant loadings from one PCA fit.

    Axes report the explained-variance percentages stored on ``PCAResult``.
    Cities far from the median score are labeled on the scatter. The
    remaining names are listed under the figure so the central group stays
    readable.
    """

    _require_columns(df, ["city"])
    result = fit_standardised_pca(df, n_components=2)
    scores = result.scores
    loadings = result.loadings
    cities = df.loc[scores.index, "city"].astype(str)

    fig = plt.figure(figsize=(13.2, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[5.2, 1.5], width_ratios=[1.35, 1])
    ax_scores = fig.add_subplot(grid[0, 0])
    ax_loadings = fig.add_subplot(grid[0, 1])
    ax_note = fig.add_subplot(grid[1, :])
    ax_note.axis("off")

    ax_scores.scatter(scores["PC1"], scores["PC2"], color="#4C72B0", s=36, zorder=3)
    ax_scores.axhline(0, color="#B0B0B0", linewidth=0.8)
    ax_scores.axvline(0, color="#B0B0B0", linewidth=0.8)
    labeled_index, other_cities = _pca_label_split(scores, cities)
    _annotate_score_labels(
        ax_scores,
        scores,
        cities.loc[labeled_index],
    )
    ax_scores.margins(0.15)
    ax_scores.set_xlabel(_component_axis_label(result, "PC1"))
    ax_scores.set_ylabel(_component_axis_label(result, "PC2"))
    ax_scores.set_title(
        "Standardised pollutant-profile snapshot\n"
        "Position compares profile shape in this sample, not severity"
    )
    ax_scores.grid(True, linestyle="--", alpha=0.35)

    _draw_loadings(ax_loadings, loadings)
    ax_note.text(0, 1, _other_cities_note(other_cities), va="top", ha="left", fontsize=9)

    _save_figure(fig, output_path)


def plot_spearman_heatmap(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the Spearman correlation of the eight pollutant columns."""

    correlation = spearman_correlation(df)
    values = correlation.to_numpy(dtype=float)
    labels = [POLLUTANT_DISPLAY_NAMES[column] for column in POLLUTANT_COLUMNS]

    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(
        "Spearman rank correlation in this snapshot\n"
        "Association among pollutant ranks for these observations only"
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            coefficient = values[row, column]
            color = "white" if abs(coefficient) >= 0.65 else "black"
            ax.text(
                column,
                row,
                f"{coefficient:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")
    _save_figure(fig, output_path)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _extreme_particulate_index(pm2_5: pd.Series, pm10: pd.Series) -> list:
    """Return row labels at a particulate extreme, in selection order.

    A row qualifies when either column is at its minimum, its maximum, or
    at or above its 90th percentile.
    """

    selected = []
    for series in (pm2_5, pm10):
        threshold = float(series.quantile(0.9))
        extremes = (series == series.min()) | (series == series.max()) | (series >= threshold)
        for index in series.index[extremes]:
            if index not in selected:
                selected.append(index)
    return selected


def _annotate_particulate_extremes(ax, df: pd.DataFrame) -> None:
    selected = _extreme_particulate_index(df["pm2_5"], df["pm10"])
    grouped: dict[tuple[float, float], list[str]] = {}
    for index in selected:
        key = (float(df.at[index, "pm2_5"]), float(df.at[index, "pm10"]))
        grouped.setdefault(key, []).append(str(df.at[index, "city"]))

    x_span = float(df["pm2_5"].max() - df["pm2_5"].min()) or 1.0
    y_span = float(df["pm10"].max() - df["pm10"].min()) or 1.0
    for (x_value, y_value), names in grouped.items():
        for offset, name in enumerate(names):
            if len(names) == 1:
                textcoords = "offset points"
                xytext = (8, 8)
            else:
                # Park shared-coordinate labels in the open band above the corner cluster.
                textcoords = "data"
                xytext = (
                    x_value + 0.03 * x_span,
                    y_value + (0.42 + offset * 0.10) * y_span,
                )
            ax.annotate(
                name,
                (x_value, y_value),
                textcoords=textcoords,
                xytext=xytext,
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.6},
            )


def _pca_label_split(
    scores: pd.DataFrame,
    cities: pd.Series,
) -> tuple[pd.Index, list[str]]:
    """Split cities into on-scatter labels and an alphabetical remainder.

    A city is drawn on the scatter when its distance from the median score
    is at or above the 66th percentile. That keeps labels on the separated
    profiles and leaves the central group listed underneath.
    """

    distance = np.hypot(
        scores["PC1"] - scores["PC1"].median(),
        scores["PC2"] - scores["PC2"].median(),
    )
    cutoff = float(distance.quantile(0.66))
    on_plot = distance >= cutoff
    labeled_index = scores.index[on_plot]
    other_cities = sorted(cities.loc[~on_plot].astype(str).tolist())
    return labeled_index, other_cities


def _annotate_score_labels(ax, scores: pd.DataFrame, cities: pd.Series) -> None:
    origin_x = float(scores["PC1"].median())
    origin_y = float(scores["PC2"].median())
    span = float(
        np.hypot(
            scores["PC1"].max() - scores["PC1"].min(),
            scores["PC2"].max() - scores["PC2"].min(),
        )
    )
    close = 0.08 * span if span else 1.0
    groups = _group_nearby_rows(scores.loc[cities.index], close)

    for group in groups:
        centroid_x = float(scores.loc[group, "PC1"].mean())
        centroid_y = float(scores.loc[group, "PC2"].mean())
        dx = centroid_x - origin_x
        dy = centroid_y - origin_y
        norm = float(np.hypot(dx, dy)) or 1.0
        outward = (18 * dx / norm, 18 * dy / norm)
        perpendicular = (-dy / norm, dx / norm)
        ordered = sorted(group, key=lambda index: str(cities.loc[index]))
        for step, index in enumerate(ordered):
            shift = step - (len(ordered) - 1) / 2
            xytext = (
                outward[0] + shift * 24 * perpendicular[0],
                outward[1] + shift * 24 * perpendicular[1],
            )
            ax.annotate(
                str(cities.loc[index]),
                (scores.at[index, "PC1"], scores.at[index, "PC2"]),
                textcoords="offset points",
                xytext=xytext,
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.6},
            )


def _group_nearby_rows(scores: pd.DataFrame, close: float) -> list[list]:
    remaining = list(scores.index)
    groups = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        grew = True
        while grew:
            grew = False
            for index in list(remaining):
                near = any(
                    _score_distance(scores, index, member) <= close for member in group
                )
                if near:
                    group.append(index)
                    remaining.remove(index)
                    grew = True
        groups.append(group)
    return groups


def _score_distance(scores: pd.DataFrame, left, right) -> float:
    return float(
        np.hypot(
            scores.at[left, "PC1"] - scores.at[right, "PC1"],
            scores.at[left, "PC2"] - scores.at[right, "PC2"],
        )
    )


def _draw_loadings(ax, loadings: pd.DataFrame) -> None:
    positions = np.arange(len(POLLUTANT_COLUMNS))
    bar_height = 0.36
    ax.barh(
        positions - bar_height / 2,
        loadings.loc[POLLUTANT_COLUMNS, "PC1"],
        height=bar_height,
        color="#4C72B0",
        label="PC1",
    )
    ax.barh(
        positions + bar_height / 2,
        loadings.loc[POLLUTANT_COLUMNS, "PC2"],
        height=bar_height,
        color="#DD8452",
        label="PC2",
    )
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels([POLLUTANT_DISPLAY_NAMES[column] for column in POLLUTANT_COLUMNS])
    ax.set_xlabel("Loading on the standardised pollutants")
    ax.set_title("Pollutant loadings")
    ax.legend(frameon=False, loc="best")
    ax.invert_yaxis()


def _other_cities_note(cities: list[str]) -> str:
    if not cities:
        return "Every city in this snapshot is labeled on the scatter."
    lines = ["Other cities in this snapshot:"]
    row: list[str] = []
    for city in cities:
        row.append(city)
        if len(row) == 8:
            lines.append(", ".join(row))
            row = []
    if row:
        lines.append(", ".join(row))
    return "\n".join(lines)


def _component_axis_label(result, component: str) -> str:
    percent = 100 * float(result.explained_variance_ratio[component])
    return f"{component} ({percent:.1f}% of standardised variance)"


def _save_figure(fig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    finally:
        plt.close(fig)
