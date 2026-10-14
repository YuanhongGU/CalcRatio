"""Fit the blue volume fraction from Hue, Saturation, and Value.

The three columns are the original color readings used for the PCA in this
folder. A linear plane and a logistic are fit on the pooled samples and inside
each experiment. The two-variable plane, ratio ~ Hue + Saturation, is fit on
the same rows so the contribution of Value can be read off a nested F test.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent.parent
PCA_DIR = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fit_model import add_normalized_color, attach_pure_pigments, fit_mixing_model, load_stage_table


FEATURES = ["Hue", "Saturation", "Value"]
PLANE_FEATURES = ["Hue", "Saturation"]
BG = "#000000"
TEXT = "#f3f4f6"
MUTED = "#9ca3af"
PINK = "#e84d85"
BLUE = "#24b8ec"
ORANGE = "#ee7954"
GRID = "#4b5563"
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
    """Parse input tables and output paths."""
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
    parser.add_argument("--output_dir", required=False, type=str, default=str(PCA_DIR / "results"),
                        help="directory for fit tables and the text summary")
    parser.add_argument("--figure_dir", required=False, type=str, default=str(PCA_DIR / "figures"),
                        help="directory for fit figures")
    parser.add_argument("--eta_clip", required=False, type=float, default=30.0,
                        help="absolute clip applied to the logistic linear predictor")
    parser.add_argument("--optimizer", required=False, type=str, default="lm",
                        help="scipy least_squares method")
    parser.add_argument("--fig_width", required=False, type=float, default=7.2,
                        help="figure width in inches")
    parser.add_argument("--fig_height", required=False, type=float, default=5.4,
                        help="figure height in inches")
    parser.add_argument("--dpi", required=False, type=int, default=160,
                        help="figure resolution")
    parser.add_argument("--digits", required=False, type=int, default=4,
                        help="decimal places in the text summary")
    return parser.parse_args()


def regression_metrics(ratio: np.ndarray, fitted: np.ndarray, n_params: int) -> dict:
    """R-squared and RMSE of a fitted blue volume fraction.

    RMSE is the root mean squared residual, without a degrees-of-freedom correction,
    matching the CalcRatio mixing model.

    :param ratio: Prepared blue volume fraction.
    :param fitted: Fitted blue volume fraction.
    :param n_params: Number of estimated coefficients, including the intercept.
    :returns: n, residual degrees of freedom, R-squared, and RMSE.
    """
    residual = fitted - ratio
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((ratio - ratio.mean()) ** 2))
    return {
        "n": int(ratio.size),
        "dof": int(ratio.size - n_params),
        "rsquared": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def group_metrics(data: pd.DataFrame, fitted: np.ndarray, scope: str, model: str) -> pd.DataFrame:
    """Score one fit inside each experiment.

    :param data: Samples aligned with ``fitted``.
    :param fitted: Fitted blue volume fraction.
    :param scope: Sample set name.
    :param model: Model name.
    :returns: One row per experiment.
    """
    rows = []
    scored = data.copy()
    scored["fitted"] = fitted
    for group, sub in scored.groupby("group", sort=True):
        metrics = regression_metrics(sub["ratio"].to_numpy(dtype=float), sub["fitted"].to_numpy(dtype=float), 0)
        rows.append({
            "scope": scope,
            "model": model,
            "group": group,
            "n": metrics["n"],
            "rsquared": metrics["rsquared"],
            "rmse": metrics["rmse"],
        })
    return pd.DataFrame(rows)


def fit_linear(data: pd.DataFrame, features: list[str]) -> dict:
    """Ordinary least squares of ratio on the given color columns.

    :param data: Samples with ``ratio`` and ``features``.
    :param features: Predictor columns, in equation order.
    :returns: Coefficients, fitted values, and the statsmodels result.
    """
    predictors = sm.add_constant(data[features], has_constant="add")
    result = sm.OLS(data["ratio"], predictors).fit()
    fitted = np.asarray(result.fittedvalues, dtype=float)
    scales = data[features].std(ddof=1).to_numpy(dtype=float)
    coef_per_sd = np.concatenate([[np.nan], result.params.to_numpy(dtype=float)[1:] * scales])
    terms = ["intercept", *features]
    coefficients = pd.DataFrame({
        "term": terms,
        "coef": result.params.to_numpy(dtype=float),
        "std_err": result.bse.to_numpy(dtype=float),
        "t": result.tvalues.to_numpy(dtype=float),
        "p": result.pvalues.to_numpy(dtype=float),
        "coef_per_sd": coef_per_sd,
    })
    metrics = regression_metrics(data["ratio"].to_numpy(dtype=float), fitted, len(terms))
    return {
        "kind": "linear",
        "features": features,
        "coefficients": coefficients,
        "fitted": fitted,
        "metrics": metrics,
        "result": result,
        "means": None,
        "scales": None,
    }


def standardize(data: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns with the sample standard deviation.

    :param data: Samples.
    :param features: Columns to scale.
    :returns: Standardized matrix, means, and standard deviations.
    :raise ValueError: a column has no variation
    """
    values = data[features].to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=1)
    if np.any(np.isclose(scales, 0.0)):
        raise ValueError("a color column has zero variance")
    return (values - means) / scales, means, scales


def logistic_ratio(standardized: np.ndarray, params: np.ndarray, eta_clip: float) -> np.ndarray:
    """Logistic of an intercept plus one coefficient per standardized column.

    :param standardized: Z-scored predictors, shape ``(n, p)``.
    :param params: Intercept followed by one slope per column.
    :param eta_clip: Absolute clip of the linear predictor.
    :returns: Predicted blue volume fraction in (0, 1).
    """
    eta = params[0] + standardized @ params[1:]
    eta = np.clip(eta, -eta_clip, eta_clip)
    return 1.0 / (1.0 + np.exp(-eta))


def fit_logistic(data: pd.DataFrame, features: list[str], eta_clip: float, optimizer: str) -> dict:
    """Least-squares logistic of ratio on standardized Hue, Saturation, and Value.

    Slopes are started from a linear regression of the logit of the ratio.
    Ratio values of 0 and 1 are moved just inside (0, 1) for that start only.

    :param data: Samples with ``ratio`` and ``features``.
    :param features: Predictor columns.
    :param eta_clip: Absolute clip of the linear predictor.
    :param optimizer: Scipy least_squares method.
    :returns: Coefficients on the standardized scale, fitted values, and scaling.
    :raise RuntimeError: the optimizer does not converge
    """
    standardized, means, scales = standardize(data, features)
    ratio = data["ratio"].to_numpy(dtype=float)
    clipped = np.clip(ratio, 1e-4, 1.0 - 1e-4)
    logit = np.log(clipped / (1.0 - clipped))
    start_fit = sm.OLS(logit, sm.add_constant(standardized, has_constant="add")).fit()
    start = np.asarray(start_fit.params, dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        return logistic_ratio(standardized, params, eta_clip) - ratio

    fitted_result = least_squares(residual, start, method=optimizer)
    if not fitted_result.success:
        raise RuntimeError(f"logistic model did not converge: {fitted_result.message}")

    fitted = logistic_ratio(standardized, fitted_result.x, eta_clip)
    resid = fitted - ratio
    terms = ["intercept", *features]
    dof = len(ratio) - len(terms)
    sigma2 = float(np.sum(resid ** 2) / dof)
    jacobian = fitted_result.jac
    covariance = sigma2 * np.linalg.inv(jacobian.T @ jacobian)
    std_err = np.sqrt(np.diag(covariance))
    t_value = fitted_result.x / std_err
    p_value = 2.0 * stats.t.sf(np.abs(t_value), dof)
    coefficients = pd.DataFrame({
        "term": terms,
        "coef": fitted_result.x,
        "std_err": std_err,
        "t": t_value,
        "p": p_value,
        "coef_per_sd": fitted_result.x,
    })
    coefficients.loc[coefficients["term"] == "intercept", "coef_per_sd"] = np.nan
    return {
        "kind": "logistic",
        "features": features,
        "coefficients": coefficients,
        "fitted": fitted,
        "metrics": regression_metrics(ratio, fitted, len(terms)),
        "result": None,
        "means": means,
        "scales": scales,
    }


def variance_inflation(data: pd.DataFrame, features: list[str], scope: str) -> pd.DataFrame:
    """Variance inflation factor of each color column.

    :param data: Samples.
    :param features: Predictor columns.
    :param scope: Sample set name.
    :returns: One row per feature.
    """
    rows = []
    for feature in features:
        others = [name for name in features if name != feature]
        others_fit = sm.OLS(data[feature], sm.add_constant(data[others], has_constant="add")).fit()
        r_squared = float(others_fit.rsquared)
        vif = np.inf if np.isclose(r_squared, 1.0) else 1.0 / (1.0 - r_squared)
        rows.append({"scope": scope, "feature": feature, "rsquared_on_others": r_squared, "vif": float(vif)})
    return pd.DataFrame(rows)


def prediction_frame(data: pd.DataFrame, fitted: np.ndarray, scope: str, model: str) -> pd.DataFrame:
    """One row per sample for a single fit.

    :param data: Samples aligned with ``fitted``.
    :param fitted: Fitted blue volume fraction.
    :param scope: Sample set name.
    :param model: Model name.
    :returns: Prediction table.
    """
    frame = data[["file", "group", "ratio", *FEATURES]].copy()
    frame.insert(0, "model", model)
    frame.insert(0, "scope", scope)
    frame["fitted"] = fitted
    frame["residual"] = fitted - data["ratio"].to_numpy(dtype=float)
    return frame


def style_axes(ax) -> None:
    """Black axes, light ticks, and a readable legend.

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


def plot_predicted(
        panels: list[tuple[pd.DataFrame, np.ndarray, str]],
        path: Path,
        fig_width: float,
        fig_height: float,
        dpi: int,
) -> None:
    """Predicted ratio against the prepared ratio.

    :param panels: Data, fitted values, and a panel title.
    :param path: Output path.
    :param fig_width: Width of one panel, in inches.
    :param fig_height: Figure height in inches.
    :param dpi: Resolution.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width * len(panels) * 0.72, fig_height), squeeze=False)
    for ax, (data, fitted, title) in zip(axes[0], panels):
        lower = min(0.0, float(np.min(fitted)), float(data["ratio"].min()))
        upper = max(1.0, float(np.max(fitted)), float(data["ratio"].max()))
        for group, sub in data.groupby("group", sort=True):
            group_key = str(group)
            ax.scatter(
                sub["ratio"],
                fitted[sub.index],
                marker=MARKERS.get(group_key, "o"),
                color=GROUP_COLORS.get(group_key, BLUE),
                s=52,
                edgecolors=TEXT,
                linewidths=0.45,
                label=GROUP_NAMES.get(group_key, group_key),
                zorder=3,
            )
        ax.plot([lower, upper], [lower, upper], color=PINK, linewidth=1.4, label="y = x", zorder=2)
        ax.set_xlabel("Prepared blue volume fraction")
        ax.set_ylabel("Fitted blue volume fraction")
        ax.set_title(title)
        ax.legend(frameon=True, loc="best", title="Batch")
        style_axes(ax)
    save_figure(fig, path, dpi)


def plot_residuals(
        panels: list[tuple[pd.DataFrame, np.ndarray, str]],
        path: Path,
        fig_width: float,
        fig_height: float,
        dpi: int,
) -> None:
    """Residuals against the fitted ratio.

    :param panels: Data, fitted values, and a panel title.
    :param path: Output path.
    :param fig_width: Width of one panel, in inches.
    :param fig_height: Figure height in inches.
    :param dpi: Resolution.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width * len(panels) * 0.72, fig_height), squeeze=False)
    for ax, (data, fitted, title) in zip(axes[0], panels):
        residual = fitted - data["ratio"].to_numpy(dtype=float)
        for group, sub in data.groupby("group", sort=True):
            group_key = str(group)
            ax.scatter(
                fitted[sub.index],
                residual[sub.index],
                marker=MARKERS.get(group_key, "o"),
                color=GROUP_COLORS.get(group_key, BLUE),
                s=52,
                edgecolors=TEXT,
                linewidths=0.45,
                label=GROUP_NAMES.get(group_key, group_key),
                zorder=3,
            )
        ax.axhline(0.0, color=PINK, linewidth=1.4, label="Zero residual", zorder=2)
        ax.set_xlabel("Fitted blue volume fraction")
        ax.set_ylabel("Residual (fitted − prepared)")
        ax.set_title(title)
        ax.legend(frameon=True, loc="best", title="Batch")
        style_axes(ax)
    save_figure(fig, path, dpi)


def signed_sum(intercept: float, slopes: list[tuple[str, float]], digits: int) -> str:
    """Format an intercept plus slopes, with a minus sign for negative terms.

    :param intercept: Intercept.
    :param slopes: ``(name, coefficient)`` pairs.
    :param digits: Decimal places.
    :returns: The right-hand side, without the response name.
    """
    parts = [f"{intercept:.{digits}f}"]
    for name, coef in slopes:
        sign = "-" if coef < 0.0 else "+"
        parts.append(f"{sign} {abs(coef):.{digits}f} * {name}")
    return " ".join(parts)


def equation_line(model_name: str, fit: dict, digits: int) -> str:
    """Write one fitted equation.

    :param model_name: Name used in the tables.
    :param fit: Output of ``fit_linear`` or ``fit_logistic``.
    :param digits: Decimal places.
    :returns: Equation text.
    """
    coef = dict(zip(fit["coefficients"]["term"], fit["coefficients"]["coef"]))
    if fit["kind"] == "linear":
        slopes = [(name, float(coef[name])) for name in fit["features"]]
        return f"{model_name}: ratio = {signed_sum(float(coef['intercept']), slopes, digits)}"

    scaled = ", ".join(
        f"{name}_z = ({name} - {mean:.{digits}f}) / {scale:.{digits}f}"
        for name, mean, scale in zip(fit["features"], fit["means"], fit["scales"])
    )
    slopes = [(f"{name}_z", float(coef[name])) for name in fit["features"]]
    linear = signed_sum(float(coef["intercept"]), slopes, digits)
    return f"{model_name}: {scaled}; ratio = 1 / (1 + exp(-({linear})))"


def format_summary(
        fits: dict,
        metrics: pd.DataFrame,
        group_rows: pd.DataFrame,
        vif: pd.DataFrame,
        value_tests: pd.DataFrame,
        digits: int,
) -> str:
    """Build the text report.

    :param fits: ``(scope, model)`` to fit dictionary.
    :param metrics: Overall R-squared and RMSE.
    :param group_rows: Per-experiment scores of each fit.
    :param vif: Variance inflation factors.
    :param value_tests: Nested F tests for adding Value.
    :param digits: Decimal places.
    :returns: Report text.
    """
    lines = [
        "Models for the blue volume fraction from Hue, Saturation, and Value.",
        "Experiment 1 uses stage 2. Experiment 2 uses its only stage.",
        "Linear models use the original units. Logistic slopes are on columns z-scored inside that sample set.",
        "coef_per_sd is the change in ratio for a one-standard-deviation change in that column.",
        "The nested F test asks whether Value improves ratio ~ Hue + Saturation.",
        "shared_logistic is the existing curve on z_hue and z_saturation, included as a reference on the pooled rows.",
        "",
    ]
    for (scope, model), fit in fits.items():
        if fit["kind"] not in {"linear", "logistic"}:
            continue
        lines.append(equation_line(f"[{scope}] {model}", fit, digits))
    lines.append("")
    lines.append("overall fit")
    for row in metrics.itertuples(index=False):
        lines.append(
            f"  [{row.scope}] {row.model}: n={row.n:d}  dof={row.dof:d}  "
            f"R-squared={row.rsquared:.{digits}f}  RMSE={row.rmse:.{digits}f}"
        )
    lines.append("")
    lines.append("fit by experiment, using the model trained on that scope")
    for row in group_rows.itertuples(index=False):
        lines.append(
            f"  [{row.scope}] {row.model} / experiment {row.group}: n={row.n:d}  "
            f"R-squared={row.rsquared:.{digits}f}  RMSE={row.rmse:.{digits}f}"
        )
    lines.append("")
    lines.append("coefficients")
    for (scope, model), fit in fits.items():
        if fit["kind"] not in {"linear", "logistic"}:
            continue
        lines.append(f"  [{scope}] {model}")
        for row in fit["coefficients"].itertuples(index=False):
            per_sd = " " if np.isnan(row.coef_per_sd) else f"  per SD {row.coef_per_sd:.{digits}f}"
            lines.append(
                f"    {row.term}: coef {row.coef:.{digits}f}  se {row.std_err:.{digits}f}  "
                f"t {row.t:.{digits}f}  p {row.p:.{digits}f}{per_sd}"
            )
    lines.append("")
    lines.append("variance inflation of Hue, Saturation, and Value")
    for row in vif.itertuples(index=False):
        lines.append(f"  [{row.scope}] {row.feature}: VIF {row.vif:.{digits}f}")
    lines.append("")
    lines.append("nested test of Value after Hue and Saturation")
    for row in value_tests.itertuples(index=False):
        lines.append(
            f"  [{row.scope}] F({row.df_diff:.0f}, {row.df_full:.0f}) = {row.f_value:.{digits}f}, "
            f"p = {row.p_value:.{digits}f}"
        )
    return "\n".join(lines).rstrip() + "\n"


def collect_fit(bucket: dict, scope: str, model: str, fit: dict) -> None:
    """Store a fit under a scope and model name.

    :param bucket: Destination mapping.
    :param scope: Sample set name.
    :param model: Model name.
    :param fit: Fit dictionary.
    """
    bucket[(scope, model)] = fit


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    experiment_1 = load_stage_table(
        args.feature_file_1, args.mapping_file_1, args.stage_1, args.group_1,
    ).reset_index(drop=True)
    experiment_2 = load_stage_table(
        args.feature_file_2, args.mapping_file_2, args.stage_2, args.group_2,
    ).reset_index(drop=True)
    pooled = pd.concat([experiment_1, experiment_2], ignore_index=True)
    scopes = {"pooled": pooled, "experiment_1": experiment_1, "experiment_2": experiment_2}

    fits = {}
    vif_frames = []
    value_tests = []
    for scope, data in scopes.items():
        linear_hs = fit_linear(data, PLANE_FEATURES)
        linear_hsv = fit_linear(data, FEATURES)
        logistic_hsv = fit_logistic(data, FEATURES, args.eta_clip, args.optimizer)
        collect_fit(fits, scope, "linear_hs", linear_hs)
        collect_fit(fits, scope, "linear_hsv", linear_hsv)
        collect_fit(fits, scope, "logistic_hsv", logistic_hsv)
        vif_frames.append(variance_inflation(data, FEATURES, scope))
        f_value, p_value, df_diff = linear_hsv["result"].compare_f_test(linear_hs["result"])
        value_tests.append({
            "scope": scope,
            "f_value": float(f_value),
            "p_value": float(p_value),
            "df_diff": float(df_diff),
            "df_full": float(linear_hsv["result"].df_resid),
        })

    reference = attach_pure_pigments(pooled, 0.0, 1.0)
    reference = add_normalized_color(reference)
    shared = fit_mixing_model(reference, args.eta_clip, 0.0, 1.0, 1.0, args.optimizer)
    fits[("pooled", "shared_logistic")] = {
        "kind": "reference",
        "features": ["z_hue", "z_saturation"],
        "coefficients": pd.DataFrame(),
        "fitted": shared["fitted"],
        "metrics": {
            "n": int(len(reference)),
            "dof": int(shared["dof"]),
            "rsquared": float(shared["rsquared"]),
            "rmse": float(shared["rmse"]),
        },
        "result": None,
        "means": None,
        "scales": None,
    }

    metric_rows = []
    group_frames = []
    prediction_frames = []
    coefficient_frames = []
    scale_rows = []
    for (scope, model), fit in fits.items():
        metric_rows.append({"scope": scope, "model": model, **fit["metrics"]})
        group_frames.append(group_metrics(scopes[scope], fit["fitted"], scope, model))
        prediction_frames.append(prediction_frame(scopes[scope], fit["fitted"], scope, model))
        if fit["kind"] in {"linear", "logistic"}:
            coefficients = fit["coefficients"].copy()
            coefficients.insert(0, "model", model)
            coefficients.insert(0, "scope", scope)
            coefficient_frames.append(coefficients)
        if fit["means"] is not None:
            for feature, mean, scale in zip(fit["features"], fit["means"], fit["scales"]):
                scale_rows.append({
                    "scope": scope,
                    "model": model,
                    "feature": feature,
                    "mean": float(mean),
                    "std": float(scale),
                })

    metrics = pd.DataFrame(metric_rows)
    group_rows = pd.concat(group_frames, ignore_index=True)
    vif = pd.concat(vif_frames, ignore_index=True)
    tests = pd.DataFrame(value_tests)
    summary = format_summary(fits, metrics, group_rows, vif, tests, args.digits)

    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    group_rows.to_csv(output_dir / "model_group_metrics.csv", index=False)
    pd.concat(coefficient_frames, ignore_index=True).to_csv(output_dir / "model_coefficients.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "model_predictions.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(output_dir / "model_standardization.csv", index=False)
    vif.to_csv(output_dir / "model_vif.csv", index=False)
    tests.to_csv(output_dir / "model_value_test.csv", index=False)
    (output_dir / "model_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    plot_predicted(
        [
            (pooled, fits[("pooled", "linear_hsv")]["fitted"], "Pooled linear: Hue + Saturation + Value"),
            (pooled, fits[("pooled", "logistic_hsv")]["fitted"], "Pooled logistic: Hue + Saturation + Value"),
        ],
        figure_dir / "model_pooled_pred.png",
        args.fig_width,
        args.fig_height,
        args.dpi,
    )
    plot_residuals(
        [
            (pooled, fits[("pooled", "linear_hsv")]["fitted"], "Pooled linear residuals"),
            (pooled, fits[("pooled", "logistic_hsv")]["fitted"], "Pooled logistic residuals"),
        ],
        figure_dir / "model_pooled_residual.png",
        args.fig_width,
        args.fig_height,
        args.dpi,
    )
    plot_predicted(
        [
            (experiment_1, fits[("experiment_1", "linear_hsv")]["fitted"], "Experiment 1 linear"),
            (experiment_2, fits[("experiment_2", "linear_hsv")]["fitted"], "Experiment 2 linear"),
        ],
        figure_dir / "model_experiment_linear_pred.png",
        args.fig_width,
        args.fig_height,
        args.dpi,
    )
    plot_predicted(
        [
            (experiment_1, fits[("experiment_1", "logistic_hsv")]["fitted"], "Experiment 1 logistic"),
            (experiment_2, fits[("experiment_2", "logistic_hsv")]["fitted"], "Experiment 2 logistic"),
        ],
        figure_dir / "model_experiment_logistic_pred.png",
        args.fig_width,
        args.fig_height,
        args.dpi,
    )
    print(f"tables saved to\n\t{output_dir}")
    print(f"figures saved to\n\t{figure_dir}")


if __name__ == "__main__":
    main()
