"""Fit one blue-yellow mixing curve shared by every phyphox experiment.

Each sample is placed between that experiment's pure yellow and pure blue.
A logistic then maps the normalized hue and saturation to the blue volume fraction.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import least_squares

from data_preprocess import circular_mean_deg, split_csv


ROOT = Path(__file__).resolve().parent.parent
PARAM_NAMES = ["intercept", "z_hue", "z_saturation"]


def parse_args() -> argparse.Namespace:
    """Parse the input and output paths."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_file_1",      required=False, type=str,   default=str(ROOT / "data" / "1_processed" / "extracted_features.npy"),
                        help="stage features of experiment 1")
    parser.add_argument("--mapping_file_1",      required=False, type=str,   default=str(ROOT / "data" / "1_settings" / "mapping.npy"),
                        help="filename-to-ratio mapping of experiment 1")
    parser.add_argument("--stage_1",             required=False, type=int,   default=2,
                        help="stage used in experiment 1")
    parser.add_argument("--group_1",             required=False, type=str,   default="1",
                        help="label of experiment 1")
    parser.add_argument("--feature_file_2",      required=False, type=str,   default=str(ROOT / "data" / "2_processed" / "extracted_features.npy"),
                        help="stage features of experiment 2")
    parser.add_argument("--mapping_file_2",      required=False, type=str,   default=str(ROOT / "data" / "2_settings" / "mapping.npy"),
                        help="filename-to-ratio mapping of experiment 2")
    parser.add_argument("--stage_2",             required=False, type=int,   default=1,
                        help="stage used in experiment 2")
    parser.add_argument("--group_2",             required=False, type=str,   default="2",
                        help="label of experiment 2")
    parser.add_argument("--output_dir",          required=False, type=str,   default=str(ROOT / "data" / "2_processed"),
                        help="directory for the combined table and fitted parameters")
    parser.add_argument("--table_filename",      required=False, type=str,   default="combined_modeling_data.pkl",
                        help="filename of the combined modeling table")
    parser.add_argument("--param_filename",      required=False, type=str,   default="mixing_model.npy",
                        help="filename of the fitted parameters")
    parser.add_argument("--yellow_ratio",        required=False, type=float, default=0.0,
                        help="blue volume fraction of the pure yellow stock")
    parser.add_argument("--blue_ratio",          required=False, type=float, default=1.0,
                        help="blue volume fraction of the pure blue stock")
    parser.add_argument("--eta_clip",            required=False, type=float, default=30.0,
                        help="absolute clip applied to the logistic linear predictor")
    parser.add_argument("--init_intercept",      required=False, type=float, default=0.0,
                        help="starting value of the logistic intercept")
    parser.add_argument("--init_z_hue",          required=False, type=float, default=1.0,
                        help="starting value of the z_hue coefficient")
    parser.add_argument("--init_z_saturation",   required=False, type=float, default=1.0,
                        help="starting value of the z_saturation coefficient")
    parser.add_argument("--optimizer",           required=False, type=str,   default="lm",
                        help="scipy least_squares method")
    parser.add_argument("--baseline_cols",       required=False, type=str,   default="Hue,Saturation",
                        help="comma-separated columns of the linear baseline")
    parser.add_argument("--hist_bin_width",      required=False, type=float, default=0.1,
                        help="width of each residual histogram bin, in ratio units")
    parser.add_argument("--grid_size",           required=False, type=int,   default=50,
                        help="points along each axis of the fitted surface")
    parser.add_argument("--fig_width",           required=False, type=float, default=8.0,
                        help="figure width in inches")
    parser.add_argument("--fig_height",          required=False, type=float, default=6.0,
                        help="figure height in inches")
    parser.add_argument("--scatter_size",        required=False, type=float, default=40.0,
                        help="marker size of the 3D scatter")
    parser.add_argument("--surface_alpha",       required=False, type=float, default=0.3,
                        help="opacity of the fitted surface")
    parser.add_argument("--surface_color",       required=False, type=str,   default="crimson",
                        help="color of the fitted surface")
    parser.add_argument("--guide_color",         required=False, type=str,   default="r",
                        help="color of residual and identity guide lines")
    parser.add_argument("--markers",             required=False, type=str,   default="o,^,s,D",
                        help="comma-separated markers, one per experiment")
    parser.add_argument("--colors",              required=False, type=str,   default="tab:blue,tab:orange,tab:green,tab:red",
                        help="comma-separated colors, one per experiment")
    parser.add_argument("--coef_digits",         required=False, type=int,   default=4,
                        help="decimal places for printed coefficients and R-squared")
    parser.add_argument("--endpoint_digits",     required=False, type=int,   default=3,
                        help="decimal places for printed pure-pigment endpoints")
    parser.add_argument("--qq_line",             required=False, type=str,   default="45",
                        help="reference line drawn by the QQ plot")

    return parser.parse_args()


def load_stage_table(feature_file: str, mapping_file: str, stage: int, group: str) -> pd.DataFrame:
    """Load Hue and Saturation for one stage and attach the blue volume fraction.

    :param feature_file: ``extracted_features.npy``.
    :param mapping_file: Filename-to-ratio mapping.
    :param stage: 1-based stage index.
    :param group: Experiment label.
    :returns: One row per file.
    """
    features = np.load(feature_file, allow_pickle=True).item()
    mapping = np.load(mapping_file, allow_pickle=True).item()

    rows = []
    for filename, item in features.items():
        if filename not in mapping:
            print(f"{filename} is missing from the mapping and was skipped")
            continue
        if stage not in item:
            print(f"{filename} has no stage {stage} and was skipped")
            continue

        stage_dict = item[stage]
        rows.append({
            "file": filename,
            "group": group,
            "Hue": float(stage_dict["Hue"]),
            "Saturation": float(stage_dict["Saturation"]),
            "Value": float(stage_dict["Value"]),
            "ratio": float(mapping[filename]),
        })

    if not rows:
        raise ValueError(f"{group} has no usable samples: {feature_file}")

    return pd.DataFrame(rows)


def attach_pure_pigments(data: pd.DataFrame, yellow_ratio: float, blue_ratio: float) -> pd.DataFrame:
    """Copy each experiment's pure yellow and pure blue readings onto every row.

    Stock concentration and camera exposure move the measured endpoints, so
    Blue and Yellow are the observed pure stocks of that experiment.

    :param data: Table with group, Hue, Saturation, and ratio.
    :param yellow_ratio: Blue volume fraction of the pure yellow stock.
    :param blue_ratio: Blue volume fraction of the pure blue stock.
    :returns: Table with Blue_Hue, Blue_Saturation, Yellow_Hue, Yellow_Saturation.
    """
    data = data.copy()
    pigment_cols = ["Blue_Hue", "Blue_Saturation", "Yellow_Hue", "Yellow_Saturation"]
    for col in pigment_cols:
        data[col] = np.nan

    for group, sub in data.groupby("group"):
        yellow = sub[np.isclose(sub["ratio"], yellow_ratio)]
        blue = sub[np.isclose(sub["ratio"], blue_ratio)]
        if yellow.empty or blue.empty:
            raise ValueError(
                f"{group} is missing pure yellow (ratio={yellow_ratio}) "
                f"or pure blue (ratio={blue_ratio})"
            )

        data.loc[sub.index, "Yellow_Hue"] = circular_mean_deg(yellow["Hue"])
        data.loc[sub.index, "Yellow_Saturation"] = float(yellow["Saturation"].mean())
        data.loc[sub.index, "Blue_Hue"] = circular_mean_deg(blue["Hue"])
        data.loc[sub.index, "Blue_Saturation"] = float(blue["Saturation"].mean())

    return data


def add_normalized_color(data: pd.DataFrame) -> pd.DataFrame:
    """Express each sample as a position between its own pure yellow and pure blue.

    ``z_hue`` is 0 at that experiment's yellow endpoint and 1 at its blue endpoint.
    Saturation is scaled in the same way. The coordinate compares a sample with
    the two stocks, so absolute Hue can differ between experiments.

    :param data: Table that already contains the blue and yellow endpoints.
    :returns: Table with z_hue and z_saturation.
    """
    data = data.copy()

    hue_span = data["Blue_Hue"] - data["Yellow_Hue"]
    sat_span = data["Blue_Saturation"] - data["Yellow_Saturation"]
    if np.any(np.isclose(hue_span, 0.0)) or np.any(np.isclose(sat_span, 0.0)):
        raise ValueError("pure blue and pure yellow have identical Hue or Saturation, so the color cannot be normalized")

    data["z_hue"] = (data["Hue"] - data["Yellow_Hue"]) / hue_span
    data["z_saturation"] = (data["Saturation"] - data["Yellow_Saturation"]) / sat_span
    return data


def mixing_ratio(
        z_hue: np.ndarray,
        z_saturation: np.ndarray,
        params: np.ndarray,
        eta_clip: float,
) -> np.ndarray:
    """Predict the blue volume fraction from the normalized color.

    The fraction lies in (0, 1) and increases as the sample moves toward blue,
    so the link is logistic:

    ratio = 1 / (1 + exp(-(a + b * z_hue + c * z_saturation)))

    :param z_hue: Hue position between pure yellow and pure blue.
    :param z_saturation: Saturation position between pure yellow and pure blue.
    :param params: ``[a, b, c]``.
    :param eta_clip: Absolute clip of the linear predictor.
    :returns: Predicted blue volume fraction.
    """
    intercept, hue_coef, sat_coef = params
    eta = intercept + hue_coef * z_hue + sat_coef * z_saturation
    eta = np.clip(eta, -eta_clip, eta_clip)
    return 1.0 / (1.0 + np.exp(-eta))


def fit_mixing_model(
        data: pd.DataFrame,
        eta_clip: float,
        init_intercept: float,
        init_z_hue: float,
        init_z_saturation: float,
        optimizer: str,
) -> dict:
    """Fit one mixing curve shared by every experiment in ``data``.

    :param data: Table with z_hue, z_saturation, and ratio.
    :param eta_clip: Absolute clip of the logistic linear predictor.
    :param init_intercept: Starting intercept.
    :param init_z_hue: Starting z_hue coefficient.
    :param init_z_saturation: Starting z_saturation coefficient.
    :param optimizer: Scipy least_squares method.
    :returns: Coefficients, standard errors, fitted values, and residuals.
    """
    z_hue = data["z_hue"].to_numpy(dtype=float)
    z_saturation = data["z_saturation"].to_numpy(dtype=float)
    ratio = data["ratio"].to_numpy(dtype=float)
    start = np.array([init_intercept, init_z_hue, init_z_saturation], dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        return mixing_ratio(z_hue, z_saturation, params, eta_clip) - ratio

    result = least_squares(residual, start, method=optimizer)
    if not result.success:
        raise RuntimeError(f"mixing model did not converge: {result.message}")

    fitted = mixing_ratio(z_hue, z_saturation, result.x, eta_clip)
    resid = fitted - ratio
    dof = len(ratio) - len(result.x)
    sigma2 = float(np.sum(resid ** 2) / dof)
    jacobian = result.jac
    cov = sigma2 * np.linalg.inv(jacobian.T @ jacobian)
    std_err = np.sqrt(np.diag(cov))
    t_value = result.x / std_err
    p_value = 2.0 * stats.t.sf(np.abs(t_value), dof)

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ratio - ratio.mean()) ** 2))

    return {
        "params": result.x,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "dof": dof,
        "fitted": fitted,
        "resid": resid,
        "rsquared": 1.0 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
    }


def linear_baseline(data: pd.DataFrame, baseline_cols: list[str]):
    """Fit the previous plane, ``ratio ~ baseline_cols``, for comparison.

    :param data: Combined color table.
    :param baseline_cols: Predictor columns.
    :returns: Statsmodels OLS result.
    """
    predictors = sm.add_constant(data[baseline_cols])
    return sm.OLS(data["ratio"], predictors).fit()


def print_model(
        data: pd.DataFrame,
        model: dict,
        baseline,
        baseline_cols: list[str],
        coef_digits: int,
        endpoint_digits: int,
) -> None:
    """Print the equation, endpoints, and error of each experiment.

    :param data: Modeling table with group labels and endpoints.
    :param model: Return value of ``fit_mixing_model``.
    :param baseline: Linear OLS result.
    :param baseline_cols: Predictor columns of the linear baseline.
    :param coef_digits: Decimal places for coefficients and R-squared.
    :param endpoint_digits: Decimal places for endpoints.
    """
    intercept, hue_coef, sat_coef = model["params"]
    print("\nmixing model")
    print("z_hue        = (Hue - Yellow_Hue) / (Blue_Hue - Yellow_Hue)")
    print("z_saturation = (Saturation - Yellow_Saturation) / (Blue_Saturation - Yellow_Saturation)")
    print(
        "ratio        = 1 / (1 + exp(-({0:.{3}f} + {1:.{3}f} * z_hue + {2:.{3}f} * z_saturation)))".format(
            intercept, hue_coef, sat_coef, coef_digits
        )
    )

    print("\nparameters")
    summary = pd.DataFrame({
        "coef": model["params"],
        "std_err": model["std_err"],
        "t": model["t_value"],
        "p": model["p_value"],
    }, index=PARAM_NAMES)
    print(summary.round(coef_digits).to_string())
    print(f"\nR-squared: {model['rsquared']:.{coef_digits}f}")
    print(f"RMSE:      {model['rmse']:.{coef_digits}f}")
    baseline_formula = " + ".join(baseline_cols)
    print(f"baseline ratio ~ {baseline_formula} R-squared: {baseline.rsquared:.{coef_digits}f}")

    print("\npure yellow / pure blue endpoints")
    endpoints = (
        data.groupby("group")[["Yellow_Hue", "Yellow_Saturation", "Blue_Hue", "Blue_Saturation"]]
        .first()
        .round(endpoint_digits)
    )
    print(endpoints.to_string())

    print("\nfit by group")
    scored = data.copy()
    scored["residual"] = model["resid"]
    for group, sub in scored.groupby("group"):
        residual = sub["residual"].to_numpy()
        ratio = sub["ratio"].to_numpy()
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((ratio - ratio.mean()) ** 2))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        print(
            f"{group}: n={len(sub):d}  "
            f"R-squared={1.0 - ss_res / ss_tot:.{coef_digits}f}  "
            f"RMSE={rmse:.{coef_digits}f}"
        )


def residual_bin_edges(resid: np.ndarray, bin_width: float) -> np.ndarray:
    """Build bin edges of a fixed width that cover every residual.

    Edges fall on multiples of ``bin_width``, so zero sits on a tick.

    :param resid: Residual values.
    :param bin_width: Width of each bin, in the same units as ``resid``.
    :returns: Increasing bin edges, including the rightmost edge.
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    values = np.asarray(resid, dtype=float)
    lowest = np.floor(np.min(values) / bin_width) * bin_width
    highest = np.ceil(np.max(values) / bin_width) * bin_width
    if highest <= lowest:
        highest = lowest + bin_width
    n_bins = int(np.round((highest - lowest) / bin_width))
    return lowest + np.arange(n_bins + 1) * bin_width


def plot_residual_histogram(resid: np.ndarray, bin_width: float):
    """Draw a residual histogram with bin-edge ticks, a y grid, and bar counts.

    :param resid: Fitted ratio minus measured ratio.
    :param bin_width: Width of each bin, in ratio units.
    :returns: Figure and axes of the histogram.
    """
    values = np.asarray(resid, dtype=float)
    edges = residual_bin_edges(values, bin_width)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    counts, _, bars = ax.hist(
        values,
        bins=edges,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#9a9a9a", zorder=0)
    ax.set_xticks(edges)
    ax.set_xlim(edges[0], edges[-1])
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8, zorder=2)

    peak = float(np.max(counts)) if len(counts) else 1.0
    ax.set_ylim(0, peak * 1.22)
    for bar, count in zip(bars, counts):
        count = int(count)
        if count == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + peak * 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1a1a1a",
        )

    ax.set_xlabel("Residual (fitted ratio − measured ratio)")
    ax.set_ylabel("Count")
    ax.set_title("Residual histogram")
    ax.text(
        0.98,
        0.98,
        f"n = {values.size}\nbin width = {bin_width:g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "linewidth": 0.6},
    )
    ax.tick_params(direction="out", length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    return fig, ax


def residual_diagnosis(
        resid: np.ndarray,
        fitted: np.ndarray,
        bin_width: float,
        guide_color: str,
) -> None:
    """Plot the residual distribution and residuals against fitted values.

    :param resid: Fitted value minus observed ratio.
    :param fitted: Predicted ratio.
    :param bin_width: Width of each residual histogram bin, in ratio units.
    :param guide_color: Color of the zero line.
    """
    resid = pd.Series(resid)
    print("\nresidual skew:", float(resid.skew()))
    print("residual kurtosis:", float(resid.kurtosis()))

    plot_residual_histogram(resid.to_numpy(), bin_width)
    plt.show()

    plt.figure()
    plt.scatter(fitted, resid)
    plt.axhline(0, color=guide_color)
    plt.xlabel("Fitted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()


def plot_fit(
        data: pd.DataFrame,
        params: np.ndarray,
        eta_clip: float,
        grid_size: int,
        fig_width: float,
        fig_height: float,
        scatter_size: float,
        surface_alpha: float,
        surface_color: str,
        markers: list[str],
        colors: list[str],
) -> None:
    """Draw both experiments and the shared surface in normalized color coordinates.

    :param data: Table with z_hue, z_saturation, ratio, and group.
    :param params: Mixing-model coefficients.
    :param eta_clip: Absolute clip of the logistic linear predictor.
    :param grid_size: Points along each surface axis.
    :param fig_width: Figure width in inches.
    :param fig_height: Figure height in inches.
    :param scatter_size: Marker size.
    :param surface_alpha: Surface opacity.
    :param surface_color: Surface color.
    :param markers: Marker cycle, one entry per experiment.
    :param colors: Color cycle, one entry per experiment.
    """
    z_hue = data["z_hue"].to_numpy(dtype=float)
    z_saturation = data["z_saturation"].to_numpy(dtype=float)

    grid_hue, grid_sat = np.meshgrid(
        np.linspace(z_hue.min(), z_hue.max(), grid_size),
        np.linspace(z_saturation.min(), z_saturation.max(), grid_size),
    )
    grid_ratio = mixing_ratio(grid_hue, grid_sat, params, eta_clip)

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111, projection="3d")
    for index, (group, sub) in enumerate(data.groupby("group")):
        ax.scatter(
            sub["z_hue"], sub["z_saturation"], sub["ratio"],
            marker=markers[index % len(markers)],
            color=colors[index % len(colors)],
            s=scatter_size,
            depthshade=False,
            label=group,
        )

    ax.plot_surface(
        grid_hue, grid_sat, grid_ratio,
        alpha=surface_alpha, color=surface_color,
        shade=False,
        antialiased=False,
    )
    ax.set_xlabel("z_hue")
    ax.set_ylabel("z_saturation")
    ax.set_zlabel("ratio")
    ax.set_title("Blue / Yellow mixing fit")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_pred_vs_actual(
        data: pd.DataFrame,
        fitted: np.ndarray,
        fig_width: float,
        fig_height: float,
        guide_color: str,
        markers: list[str],
        colors: list[str],
) -> None:
    """Plot predicted blue volume fraction against the measured fraction.

    :param data: Table with group and ratio.
    :param fitted: Predicted ratio, aligned with ``data``.
    :param fig_width: Figure width in inches.
    :param fig_height: Figure height in inches.
    :param guide_color: Color of the identity line.
    :param markers: Marker cycle, one entry per experiment.
    :param colors: Color cycle, one entry per experiment.
    """
    plt.figure(figsize=(fig_width, fig_height))
    for index, (group, sub) in enumerate(data.groupby("group")):
        plt.scatter(
            sub["ratio"], fitted[sub.index],
            marker=markers[index % len(markers)],
            color=colors[index % len(colors)],
            label=group,
        )
    plt.plot([0, 1], [0, 1], color=guide_color)
    plt.xlabel("actual ratio")
    plt.ylabel("predicted ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    baseline_cols = split_csv(args.baseline_cols)
    markers = split_csv(args.markers)
    colors = split_csv(args.colors)

    experiment_1 = load_stage_table(args.feature_file_1, args.mapping_file_1, args.stage_1, args.group_1)
    experiment_2 = load_stage_table(args.feature_file_2, args.mapping_file_2, args.stage_2, args.group_2)
    data = pd.concat([experiment_1, experiment_2], ignore_index=True)
    data = attach_pure_pigments(data, args.yellow_ratio, args.blue_ratio)
    data = add_normalized_color(data)

    model = fit_mixing_model(
        data,
        args.eta_clip,
        args.init_intercept,
        args.init_z_hue,
        args.init_z_saturation,
        args.optimizer,
    )
    baseline = linear_baseline(data, baseline_cols)
    print_model(data, model, baseline, baseline_cols, args.coef_digits, args.endpoint_digits)

    os.makedirs(args.output_dir, exist_ok=True)
    table_file = os.path.join(args.output_dir, args.table_filename)
    param_file = os.path.join(args.output_dir, args.param_filename)
    data = data.copy()
    data["ratio_hat"] = model["fitted"]
    data.to_pickle(table_file)
    np.save(param_file, {
        "params": model["params"],
        "param_names": PARAM_NAMES,
        "stage_1": args.stage_1,
        "stage_2": args.stage_2,
    })
    print(f"\nmodeling table saved to\n\t{table_file}")
    print(f"parameters saved to\n\t{param_file}")

    residual_diagnosis(model["resid"], model["fitted"], args.hist_bin_width, args.guide_color)
    plot_fit(
        data, model["params"], args.eta_clip, args.grid_size,
        args.fig_width, args.fig_height, args.scatter_size,
        args.surface_alpha, args.surface_color, markers, colors,
    )
    plot_pred_vs_actual(
        data, model["fitted"], args.fig_width, args.fig_height,
        args.guide_color, markers, colors,
    )

    sm.qqplot(model["resid"], line=args.qq_line)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
