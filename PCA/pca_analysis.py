"""Principal component analysis of the original Hue, Saturation, and Value readings.

Each phyphox file is one sample. The color is the stage mean already used by the
mixing model: stage 2 for experiment 1 and stage 1 for experiment 2. These are
the original color coordinates, before each batch is rescaled by its pure yellow
and pure blue. Camera settings are left out, because aperture, shutter, and ISO
are acquisition choices and would otherwise separate the two experiments by themselves.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent.parent
PCA_DIR = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_preprocess import split_csv
from fit_model import load_stage_table


# Software wiki mosaic palette. Low blue fraction is lime, high is blue.
BG = "#000000"
TEXT = "#f3f4f6"
MUTED = "#9ca3af"
PINK = "#e84d85"
PURPLE = "#9b6ca8"
LIME = "#c6da58"
BLUE = "#24b8ec"
ORANGE = "#ee7954"
GRID = "#4b5563"
PALETTE = [BLUE, LIME, ORANGE, PINK, PURPLE]
RATIO_CMAP = LinearSegmentedColormap.from_list(
    "mosaic_ratio", [LIME, PURPLE, BLUE],
)
MARKERS = {"1": "o", "2": "^"}
GROUP_NAMES = {"1": "Experiment 1", "2": "Experiment 2"}
GROUP_COLORS = {"1": BLUE, "2": ORANGE}


def apply_dark_theme() -> None:
    """Black figure with wiki text, axes, and series colors."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": MUTED,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "grid.color": GRID,
        "legend.facecolor": "#111111",
        "legend.edgecolor": MUTED,
        "legend.labelcolor": TEXT,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })


apply_dark_theme()


def parse_args() -> argparse.Namespace:
    """Parse input tables, feature columns, and output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_file_1", required=False, type=str,
                        default=str(ROOT / "data" / "1_processed" / "extracted_features.npy"),
                        help="stage features of experiment 1")
    parser.add_argument("--mapping_file_1", required=False, type=str,
                        default=str(ROOT / "data" / "1_settings" / "mapping.npy"),
                        help="filename-to-ratio mapping of experiment 1")
    parser.add_argument("--stage_1", required=False, type=int, default=2,
                        help="stage used in experiment 1")
    parser.add_argument("--group_1", required=False, type=str, default="1",
                        help="label of experiment 1")
    parser.add_argument("--feature_file_2", required=False, type=str,
                        default=str(ROOT / "data" / "2_processed" / "extracted_features.npy"),
                        help="stage features of experiment 2")
    parser.add_argument("--mapping_file_2", required=False, type=str,
                        default=str(ROOT / "data" / "2_settings" / "mapping.npy"),
                        help="filename-to-ratio mapping of experiment 2")
    parser.add_argument("--stage_2", required=False, type=int, default=1,
                        help="stage used in experiment 2")
    parser.add_argument("--group_2", required=False, type=str, default="2",
                        help="label of experiment 2")
    parser.add_argument("--features", required=False, type=str, default="Hue,Saturation,Value",
                        help="comma-separated original color columns")
    parser.add_argument("--output_dir", required=False, type=str, default=str(PCA_DIR / "results"),
                        help="directory for tables and the text summary")
    parser.add_argument("--figure_dir", required=False, type=str, default=str(PCA_DIR / "figures"),
                        help="directory for PCA figures")
    parser.add_argument("--fig_width", required=False, type=float, default=7.2,
                        help="figure width in inches")
    parser.add_argument("--fig_height", required=False, type=float, default=5.4,
                        help="figure height in inches")
    parser.add_argument("--dpi", required=False, type=int, default=160,
                        help="figure resolution")
    parser.add_argument("--digits", required=False, type=int, default=4,
                        help="decimal places in the text summary")
    return parser.parse_args()


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column with the sample standard deviation.

    :param values: Observation matrix, shape ``(n, p)``.
    :returns: Standardized matrix, column means, and column standard deviations.
    :raise ValueError: a column has no variation
    """
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=1)
    if np.any(np.isclose(scales, 0.0)):
        raise ValueError("a feature has zero variance and cannot be standardized")
    return (values - means) / scales, means, scales


def fit_pca(standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a standardized matrix by singular value decomposition.

    Scores are the coordinates of each sample. Loadings are the unit-length
    eigenvectors of the sample covariance. Eigenvalues use the ``n - 1`` divisor,
    matching the sample covariance of the standardized columns.

    :param standardized: Z-scored matrix, shape ``(n, p)``.
    :returns: Scores ``(n, k)``, loadings ``(p, k)``, and eigenvalues ``(k,)``.
    """
    n_rows = standardized.shape[0]
    u_matrix, singular, vt_matrix = np.linalg.svd(standardized, full_matrices=False)
    scores = u_matrix * singular
    loadings = vt_matrix.T
    eigenvalues = (singular ** 2) / (n_rows - 1)
    return scores, loadings, eigenvalues


def orient_axes(scores: np.ndarray, loadings: np.ndarray, ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose the sign of each axis so the result does not flip between runs.

    The first component is pointed so that it increases with the blue volume
    fraction. Later components are pointed so their largest loading is positive.
    The sign of a principal component is otherwise arbitrary.

    :param scores: Sample coordinates.
    :param loadings: Variable eigenvectors.
    :param ratio: Blue volume fraction, aligned with the rows of ``scores``.
    :returns: Scores and loadings with fixed signs.
    """
    scores = scores.copy()
    loadings = loadings.copy()
    if pearson(scores[:, 0], ratio) < 0.0:
        scores[:, 0] *= -1.0
        loadings[:, 0] *= -1.0
    for axis in range(1, scores.shape[1]):
        pivot = int(np.argmax(np.abs(loadings[:, axis])))
        if loadings[pivot, axis] < 0.0:
            scores[:, axis] *= -1.0
            loadings[:, axis] *= -1.0
    return scores, loadings


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation of two vectors.

    :param left: First vector.
    :param right: Second vector.
    :returns: Correlation, or NaN when either vector has no variation.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left - left.mean()
    right = right - right.mean()
    denom = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    if np.isclose(denom, 0.0):
        return float("nan")
    return float(np.sum(left * right) / denom)


def analyze(data: pd.DataFrame, features: list[str], scope: str) -> dict:
    """Run one PCA and collect tables for that set of samples.

    :param data: Samples with the feature columns, ``ratio``, ``file``, and ``group``.
    :param features: Columns entered into the decomposition.
    :param scope: Name used in the saved tables and figure filenames.
    :returns: Scores, loadings, variance, and correlations for this scope.
    """
    values = data[features].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"{scope} contains missing color values")

    standardized, means, scales = standardize(values)
    scores, loadings, eigenvalues = fit_pca(standardized)
    scores, loadings = orient_axes(scores, loadings, data["ratio"].to_numpy(dtype=float))

    n_components = scores.shape[1]
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    component_names = [f"PC{index + 1}" for index in range(n_components)]

    score_table = data[["file", "group", "ratio", *features]].copy()
    score_table.insert(0, "scope", scope)
    for index, name in enumerate(component_names):
        score_table[name] = scores[:, index]

    loading_rows = []
    for feature_index, feature in enumerate(features):
        for component_index, name in enumerate(component_names):
            eigenvector = float(loadings[feature_index, component_index])
            loading_rows.append({
                "scope": scope,
                "feature": feature,
                "component": name,
                "eigenvector": eigenvector,
                "correlation": eigenvector * np.sqrt(eigenvalues[component_index]),
            })

    variance_table = pd.DataFrame({
        "scope": scope,
        "component": component_names,
        "eigenvalue": eigenvalues,
        "explained_ratio": explained,
        "cumulative_ratio": cumulative,
    })
    ratio_rows = [
        {"scope": scope, "component": name, "pearson_r_with_ratio": pearson(scores[:, index], data["ratio"].to_numpy(dtype=float))}
        for index, name in enumerate(component_names)
    ]

    feature_rows = []
    for feature, mean, scale in zip(features, means, scales):
        column = data[feature].to_numpy(dtype=float)
        feature_rows.append({
            "scope": scope,
            "feature": feature,
            "mean": float(mean),
            "std": float(scale),
            "min": float(column.min()),
            "max": float(column.max()),
        })

    correlation_rows = []
    for left_index, left_name in enumerate(features):
        for right_index, right_name in enumerate(features):
            if right_index <= left_index:
                continue
            correlation_rows.append({
                "scope": scope,
                "feature_1": left_name,
                "feature_2": right_name,
                "pearson_r": pearson(values[:, left_index], values[:, right_index]),
            })

    return {
        "scope": scope,
        "data": data.reset_index(drop=True),
        "features": features,
        "scores": scores,
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "explained": explained,
        "score_table": score_table,
        "loading_table": pd.DataFrame(loading_rows),
        "variance_table": variance_table,
        "ratio_table": pd.DataFrame(ratio_rows),
        "feature_table": pd.DataFrame(feature_rows),
        "correlation_table": pd.DataFrame(correlation_rows),
    }


def style_axes(ax) -> None:
    """Black axes, light ticks, and a grid that stays readable.

    :param ax: Matplotlib axes.
    """
    ax.set_facecolor(BG)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    legend = ax.get_legend()
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor("#111111")
        frame.set_edgecolor(MUTED)
        for text in legend.get_texts():
            text.set_color(TEXT)


def style_colorbar(colorbar) -> None:
    """Light colorbar ticks and label on the black figure."""
    colorbar.outline.set_edgecolor(MUTED)
    colorbar.ax.yaxis.label.set_color(TEXT)
    colorbar.ax.tick_params(colors=TEXT)


def save_figure(fig, path: Path, dpi: int) -> None:
    """Write one figure and close it.

    :param fig: Matplotlib figure.
    :param path: Output path.
    :param dpi: Resolution.
    """
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)


def scatter_groups(ax, data: pd.DataFrame, x: np.ndarray, y: np.ndarray, color_by_ratio: bool) -> None:
    """Draw one marker series per experiment.

    :param ax: Matplotlib axes.
    :param data: Samples aligned with ``x`` and ``y``.
    :param x: Horizontal coordinates.
    :param y: Vertical coordinates.
    :param color_by_ratio: Color points by blue volume fraction when true, otherwise by experiment.
    """
    for group, sub in data.groupby("group", sort=True):
        group_key = str(group)
        if color_by_ratio:
            colors = sub["ratio"]
            cmap = RATIO_CMAP
            vmin, vmax = 0.0, 1.0
            facecolor = None
        else:
            colors = GROUP_COLORS.get(group_key, BLUE)
            cmap = None
            vmin, vmax = None, None
            facecolor = colors
        ax.scatter(
            x[sub.index],
            y[sub.index],
            c=facecolor if facecolor is not None else colors,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=MARKERS.get(group_key, "o"),
            s=52,
            edgecolors=TEXT,
            linewidths=0.45,
            label=GROUP_NAMES.get(group_key, group_key),
            zorder=2,
        )


def plot_scree(result: dict, figure_dir: Path, fig_width: float, fig_height: float, dpi: int) -> None:
    """Bar chart of variance explained by each component.

    :param result: Output of ``analyze``.
    :param figure_dir: Directory for the PNG file.
    :param fig_width: Figure width in inches.
    :param fig_height: Figure height in inches.
    :param dpi: Resolution.
    """
    explained = result["explained"] * 100.0
    labels = [f"PC{index + 1}" for index in range(explained.size)]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    positions = np.arange(explained.size)
    ax.bar(positions, explained, color=BLUE, width=0.72, label="This component")
    ax.plot(
        positions, np.cumsum(explained), color=ORANGE, marker="o",
        markersize=7, markeredgecolor=TEXT, markeredgewidth=0.6,
        linewidth=1.8, label="Cumulative",
    )
    for position, value in zip(positions, explained):
        ax.text(
            position, value - 2.2, f"{value:.1f}%",
            ha="center", va="top", color=BG, fontsize=9, fontweight="bold",
        )
    ax.legend(frameon=True, loc="center right")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, 118)
    ax.set_title(f"{result['scope']}: variance explained")
    style_axes(ax)
    save_figure(fig, figure_dir / f"{result['scope']}_scree.png", dpi)


def plot_scores(result: dict, figure_dir: Path, fig_width: float, fig_height: float, dpi: int) -> None:
    """PC1 against PC2, colored by the prepared blue volume fraction.

    :param result: Output of ``analyze``.
    :param figure_dir: Directory for the PNG file.
    :param fig_width: Figure width in inches.
    :param fig_height: Figure height in inches.
    :param dpi: Resolution.
    """
    data = result["data"]
    scores = result["scores"]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scatter_groups(ax, data, scores[:, 0], scores[:, 1], color_by_ratio=True)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=RATIO_CMAP, norm=plt.Normalize(0.0, 1.0)), ax=ax)
    colorbar.set_label("Blue volume fraction\n(0 yellow, 1 blue)")
    style_colorbar(colorbar)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({result['explained'][0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({result['explained'][1] * 100:.1f}% of variance)")
    ax.set_title(f"{result['scope']}: sample scores")
    if data["group"].nunique() > 1:
        ax.legend(frameon=True, loc="best", title="Batch")
    style_axes(ax)
    save_figure(fig, figure_dir / f"{result['scope']}_scores.png", dpi)


def plot_biplot(result: dict, figure_dir: Path, fig_width: float, fig_height: float, dpi: int) -> None:
    """Scores with loading arrows for the original color variables.

    Arrows are scaled to the score cloud so both layers are visible. The
    eigenvector values themselves are in the loadings table.

    :param result: Output of ``analyze``.
    :param figure_dir: Directory for the PNG file.
    :param fig_width: Figure width in inches.
    :param fig_height: Figure height in inches.
    :param dpi: Resolution.
    """
    data = result["data"]
    scores = result["scores"][:, :2]
    loadings = result["loadings"][:, :2]
    score_span = np.max(np.abs(scores))
    loading_span = np.max(np.abs(loadings))
    arrow_scale = 0.75 * score_span / loading_span

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scatter_groups(ax, data, scores[:, 0], scores[:, 1], color_by_ratio=True)
    label_x = []
    label_y = []
    for feature_index, feature in enumerate(result["features"]):
        x_end = float(loadings[feature_index, 0] * arrow_scale)
        y_end = float(loadings[feature_index, 1] * arrow_scale)
        ax.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": PINK, "lw": 1.8},
            zorder=3,
        )
        text_x = x_end * 1.06
        text_y = y_end * 1.06
        label_x.append(text_x)
        label_y.append(text_y)
        ax.text(
            text_x,
            text_y,
            feature,
            color=PINK,
            fontsize=11,
            fontweight="bold",
            ha="left" if text_x >= 0 else "right",
            va="bottom" if text_y >= 0 else "top",
            zorder=4,
        )
    all_x = np.concatenate([scores[:, 0], label_x, [0.0]])
    all_y = np.concatenate([scores[:, 1], label_y, [0.0]])
    span_x = max(all_x.max() - all_x.min(), 1.0)
    span_y = max(all_y.max() - all_y.min(), 1.0)
    ax.set_xlim(all_x.min() - 0.12 * span_x, all_x.max() + 0.12 * span_x)
    ax.set_ylim(all_y.min() - 0.12 * span_y, all_y.max() + 0.12 * span_y)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=RATIO_CMAP, norm=plt.Normalize(0.0, 1.0)), ax=ax)
    colorbar.set_label("Blue volume fraction\n(0 yellow, 1 blue)")
    style_colorbar(colorbar)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({result['explained'][0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({result['explained'][1] * 100:.1f}% of variance)")
    ax.set_title(f"{result['scope']}: scores and variable loadings")
    if data["group"].nunique() > 1:
        ax.legend(frameon=True, loc="best", title="Batch")
    style_axes(ax)
    save_figure(fig, figure_dir / f"{result['scope']}_biplot.png", dpi)


def plot_loadings(result: dict, figure_dir: Path, fig_width: float, fig_height: float, dpi: int) -> None:
    """Grouped bars of the variable loadings.

    :param result: Output of ``analyze``.
    :param figure_dir: Directory for the PNG file.
    :param fig_width: Figure width in inches.
    :param fig_height: Figure height in inches.
    :param dpi: Resolution.
    """
    loadings = result["loadings"]
    features = result["features"]
    n_components = loadings.shape[1]
    positions = np.arange(len(features))
    width = 0.8 / n_components
    colors = PALETTE
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for component in range(n_components):
        offset = (component - (n_components - 1) / 2) * width
        ax.bar(
            positions + offset,
            loadings[:, component],
            width=width * 0.92,
            color=colors[component % len(colors)],
            label=f"PC{component + 1}",
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(features)
    ax.set_ylabel("Loading on the original color variable")
    ax.set_title(f"{result['scope']}: variable loadings")
    ax.legend(frameon=True, title="Component")
    style_axes(ax)
    save_figure(fig, figure_dir / f"{result['scope']}_loadings.png", dpi)


def plot_pc_vs_ratio(result: dict, figure_dir: Path, fig_width: float, fig_height: float, dpi: int) -> None:
    """PC1 and PC2 against the prepared blue volume fraction.

    :param result: Output of ``analyze``.
    :param figure_dir: Directory for the PNG file.
    :param fig_width: Figure width in inches.
    :param fig_height: Figure height in inches.
    :param dpi: Resolution.
    """
    data = result["data"]
    scores = result["scores"]
    fig, axes = plt.subplots(1, 2, figsize=(fig_width * 1.45, fig_height), sharey=False)
    for axis, component in enumerate((0, 1)):
        ax = axes[axis]
        scatter_groups(ax, data, data["ratio"].to_numpy(dtype=float), scores[:, component], color_by_ratio=False)
        correlation = pearson(scores[:, component], data["ratio"].to_numpy(dtype=float))
        ax.set_xlabel("Prepared blue volume fraction")
        ax.set_ylabel(f"PC{component + 1} score")
        ax.set_title(f"PC{component + 1} versus ratio (r = {correlation:.3f})")
        ax.set_xlim(-0.05, 1.05)
        style_axes(ax)
    if data["group"].nunique() > 1:
        axes[0].legend(frameon=True, loc="best", title="Batch")
        style_axes(axes[0])
    save_figure(fig, figure_dir / f"{result['scope']}_pc_vs_ratio.png", dpi)


def format_summary(results: list[dict], digits: int) -> str:
    """Build the text report from every scope.

    :param results: One ``analyze`` dictionary per sample set.
    :param digits: Decimal places for printed numbers.
    :returns: Report text.
    """
    lines = [
        "Principal component analysis of the original color readings.",
        "Features are Hue (degrees), Saturation, and Value.",
        "Each column is z-scored inside the sample set before the decomposition.",
        "PC1 is oriented so that it increases with the blue volume fraction.",
        "Experiment 1 uses stage 2. Experiment 2 uses its only stage.",
        "Luma and Luminance are recorded only in experiment 1, so they are not in the shared feature set.",
        "Shutter speed, aperture, and ISO are acquisition settings and are not included.",
        "",
    ]
    for result in results:
        data = result["data"]
        lines.append(f"[{result['scope']}] n = {len(data)}")
        hue = data["Hue"]
        lines.append(
            f"  Hue range: {hue.min():.{digits}f} to {hue.max():.{digits}f} deg "
            f"(span {hue.max() - hue.min():.{digits}f})"
        )
        lines.append("  feature mean (std):")
        for row in result["feature_table"].itertuples(index=False):
            lines.append(f"    {row.feature}: {row.mean:.{digits}f} ({row.std:.{digits}f})")
        lines.append("  correlations among features:")
        for row in result["correlation_table"].itertuples(index=False):
            lines.append(f"    {row.feature_1} vs {row.feature_2}: {row.pearson_r:.{digits}f}")
        lines.append("  variance explained:")
        for row in result["variance_table"].itertuples(index=False):
            lines.append(
                f"    {row.component}: eigenvalue {row.eigenvalue:.{digits}f}, "
                f"{row.explained_ratio * 100:.{digits}f}%, "
                f"cumulative {row.cumulative_ratio * 100:.{digits}f}%"
            )
        lines.append("  loadings (eigenvector):")
        loading_wide = result["loading_table"].pivot(index="feature", columns="component", values="eigenvector")
        loading_wide = loading_wide.reindex(result["features"])
        for feature, row in loading_wide.iterrows():
            values = ", ".join(f"{name} {row[name]:.{digits}f}" for name in loading_wide.columns)
            lines.append(f"    {feature}: {values}")
        lines.append("  correlation with blue volume fraction:")
        for row in result["ratio_table"].itertuples(index=False):
            lines.append(f"    {row.component}: r = {row.pearson_r_with_ratio:.{digits}f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    features = split_csv(args.features)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    experiment_1 = load_stage_table(args.feature_file_1, args.mapping_file_1, args.stage_1, args.group_1)
    experiment_2 = load_stage_table(args.feature_file_2, args.mapping_file_2, args.stage_2, args.group_2)
    pooled = pd.concat([experiment_1, experiment_2], ignore_index=True)

    results = [
        analyze(pooled, features, "pooled"),
        analyze(experiment_1, features, "experiment_1"),
        analyze(experiment_2, features, "experiment_2"),
    ]

    pd.concat([item["score_table"] for item in results], ignore_index=True).to_csv(output_dir / "scores.csv", index=False)
    pd.concat([item["loading_table"] for item in results], ignore_index=True).to_csv(output_dir / "loadings.csv", index=False)
    pd.concat([item["variance_table"] for item in results], ignore_index=True).to_csv(output_dir / "variance.csv", index=False)
    pd.concat([item["ratio_table"] for item in results], ignore_index=True).to_csv(output_dir / "pc_ratio_correlation.csv", index=False)
    pd.concat([item["feature_table"] for item in results], ignore_index=True).to_csv(output_dir / "feature_summary.csv", index=False)
    pd.concat([item["correlation_table"] for item in results], ignore_index=True).to_csv(output_dir / "feature_correlation.csv", index=False)

    summary = format_summary(results, args.digits)
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    for result in results:
        plot_scree(result, figure_dir, args.fig_width, args.fig_height, args.dpi)
        plot_scores(result, figure_dir, args.fig_width, args.fig_height, args.dpi)
        plot_biplot(result, figure_dir, args.fig_width, args.fig_height, args.dpi)
        plot_loadings(result, figure_dir, args.fig_width, args.fig_height, args.dpi)
        plot_pc_vs_ratio(result, figure_dir, args.fig_width, args.fig_height, args.dpi)

    print(f"tables saved to\n\t{output_dir}")
    print(f"figures saved to\n\t{figure_dir}")


if __name__ == "__main__":
    main()
