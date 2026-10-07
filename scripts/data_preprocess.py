"""Turn phyphox Excel exports into stage features joined to the blue volume fraction."""

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parent.parent


# Rename these phyphox export headers. The names belong to the device schema, not to one run.
COL_MAP = {
    "t": "t",
    "Luma": "Luma",
    "Luminance": "Luminance",
    "Hue (°)": "Hue",
    "hue (°)": "Hue",
    "hue": "Hue",
    "Saturation": "Saturation",
    "saturation": "Saturation",
    "Value": "Value",
    "value": "Value",
    "Shutter Speed (s)": "Shutter_Speed",
    "Aperture (f/N)": "Aperture",
    "ISO": "ISO",
}


def split_csv(text: str) -> list[str]:
    """Split a comma-separated CLI value into stripped, non-empty items.

    :param text: Raw argument text.
    :returns: List of items.
    """
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Parse the input and output paths."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir",   required=False, type=str,   default=str(ROOT / "data" / "2_original"),
                        help="directory of phyphox Excel files")
    parser.add_argument("-o", "--output_dir",  required=False, type=str,   default=str(ROOT / "data" / "2_processed"),
                        help="directory for extracted features and the regression table")
    parser.add_argument("-n", "--n_stages",    required=False, type=int,   default=1,
                        help="number of stages to extract")
    parser.add_argument("--dict_file",         required=False, type=str,   default=str(ROOT / "data" / "2_settings" / "mapping.npy"),
                        help="filename-to-ratio mapping")
    parser.add_argument("--by_col",            required=False, type=str,   default="Hue",
                        help="column used to detect stage boundaries")
    parser.add_argument("--threshold",         required=False, type=int,   default=50,
                        help="minimum number of samples in a plateau")
    parser.add_argument("--rel_tol",           required=False, type=float, default=0.2,
                        help="relative gap between the running mean and the neighbor mean that ends a plateau")
    parser.add_argument("--abs_mean_level",    required=False, type=float, default=100.0,
                        help="running means below this level are compared with an absolute gap")
    parser.add_argument("--abs_tol",           required=False, type=float, default=20.0,
                        help="absolute gap that ends a plateau when the running mean is below abs_mean_level")
    parser.add_argument("--n_neighbors",       required=False, type=int,   default=50,
                        help="number of following samples used for the neighbor mean")
    parser.add_argument("--key_stage",         required=False, type=int,   default=1,
                        help="stage joined to the blue volume fraction (1-based)")
    parser.add_argument("--metrics",           required=False, type=str,   default="Hue,Saturation,Value",
                        help="comma-separated color columns to average inside each stage")
    parser.add_argument("--const_cols",        required=False, type=str,   default="Shutter_Speed,Aperture,ISO",
                        help="comma-separated camera columns copied from the first row")
    parser.add_argument("--circular_col",      required=False, type=str,   default="Hue",
                        help="metric averaged on the circle instead of arithmetically")
    parser.add_argument("--sheet_index",       required=False, type=int,   default=0,
                        help="Excel sheet index to read")
    parser.add_argument("--feature_filename",  required=False, type=str,   default="extracted_features.npy",
                        help="filename of the per-stage feature dictionary")
    parser.add_argument("--table_filename",    required=False, type=str,   default="processed_data.pkl",
                        help="filename of the regression table")
    parser.add_argument("--min_stage_len",     required=False, type=int,   default=1,
                        help="minimum divisor when a detected stage is empty")

    return parser.parse_args()


def circular_mean_deg(df_data: pd.DataFrame) -> float:
    """Circular mean of hue angles, in degrees on [0, 360).

    Each angle is mapped to a unit vector, the vectors are averaged, and the
    resulting direction is converted back to degrees.

    :param df_data: Hue samples in degrees.
    :returns: Circular mean in [0, 360), or NaN when no numeric sample is present.
    """
    deg_data = pd.to_numeric(df_data, errors="coerce").dropna()
    if len(deg_data) == 0:
        return np.nan

    rad = np.deg2rad(deg_data.values)
    sin_mean = np.sin(rad).mean()
    cos_mean = np.cos(rad).mean()
    rad_mean = np.arctan2(sin_mean, cos_mean)
    deg_mean = np.rad2deg(rad_mean)
    deg_mean = (deg_mean + 360.0) % 360.0

    return float(deg_mean)


def analysis_stage(
        df: pd.DataFrame,
        n_stages: int,
        by_col: str,
        n_neighbors: int,
        threshold: int,
        rel_tol: float,
        abs_mean_level: float,
        abs_tol: float,
) -> dict:
    """Find the index range of each plateau in ``by_col``.

    A stretch is kept as a stage after its running mean stays stable for at
    least ``threshold`` samples. The next samples end that stage when they
    jump by more than ``abs_tol`` (low means) or ``rel_tol`` (high means).

    :param df: One phyphox table.
    :param n_stages: Number of stages to record.
    :param by_col: Column used to detect boundaries.
    :param n_neighbors: Length of the look-ahead window.
    :param threshold: Minimum plateau length.
    :param rel_tol: Relative gap used when the running mean is at least abs_mean_level.
    :param abs_mean_level: Level below which abs_tol is used.
    :param abs_tol: Absolute gap used for low running means.
    :returns: ``{stage: (start, end)}`` with 1-based stage numbers.
    """
    df         = df.copy()
    df[by_col] = pd.to_numeric(df[by_col], errors="coerce")
    valid      = df[by_col].dropna()

    if len(valid) == 0:
        return {}

    valid = valid.tolist()

    # Record the start and end of each closed stage.
    stage_info = {}

    # Count stages that have already closed.
    cnt_stage = 0

    # Mark the start of the current candidate plateau.
    tot_stt = 0

    for idx in range(len(valid)):
        if idx < tot_stt:
            continue

        # Average from the candidate start through the current sample.
        n_tot    = idx - tot_stt + 1
        tot_mean = sum(valid[ tot_stt : idx+1 ]) / n_tot

        # Average the following neighbor window.
        end  = idx + n_neighbors
        end  = min(end, len(valid))
        n_kn = end - idx + 1
        kn_mean  = sum(valid[ idx : end+1 ]) / n_kn

        # Low signals are compared on an absolute scale. High signals use a relative gap.
        if tot_mean < abs_mean_level:
            diff     = abs(kn_mean - tot_mean)
            is_shift = diff > abs_tol
        else:
            is_shift = abs(kn_mean - tot_mean) / abs(tot_mean) > rel_tol

        if is_shift:
            # A short run is not a plateau. Restart after the neighbor window.
            if idx - tot_stt < threshold:
                tot_stt = end
                continue

            if cnt_stage == n_stages:
                break

            cnt_stage += 1
            stage_info[cnt_stage] = (tot_stt, idx)
            tot_stt = end

    if cnt_stage == 0:
        for stage_idx in range(n_stages+1):
            stage_info[stage_idx] = (tot_stt, len(valid))
    elif cnt_stage < n_stages:
        for idx in range(cnt_stage+1, n_stages+1):
            stage_info[idx] = stage_info[cnt_stage]

    return stage_info


def extract_by_stage(
        df: pd.DataFrame,
        n_stages: int,
        by_col: str,
        stage_info: dict,
        metrics: list[str],
        circular_col: str,
        const_cols: list[str],
        min_stage_len: int,
) -> dict:
    """Average each requested metric on every detected stage.

    ``circular_col`` uses a circular mean. Every other metric uses an arithmetic mean.
    Camera columns are copied from the first row.

    :param df: Renamed phyphox table.
    :param n_stages: Number of stages.
    :param by_col: Column whose non-missing rows define the stage index.
    :param stage_info: ``{stage: (start, end)}``.
    :param metrics: Color columns to average.
    :param circular_col: Column averaged on the circle.
    :param const_cols: Camera columns copied from the first row.
    :param min_stage_len: Minimum divisor for an empty stage.
    :returns: Stage dictionaries plus the camera columns.
    """
    df = df.copy()
    df = df[df[by_col].notna()].reset_index(drop=True)

    result = {}
    for stage in range(1, n_stages+1):
        stt = stage_info[stage][0]
        end = stage_info[stage][1]
        stage_len = max(end-stt, min_stage_len)

        stage_result = {}
        for metric in metrics:
            if metric == circular_col:
                stage_result[metric] = circular_mean_deg(df[metric][stt:end])
            else:
                stage_result[metric] = float(sum(df[metric].tolist()[stt:end])) / stage_len

        result[stage] = stage_result

    for col in const_cols:
        result[col] = df[col][0]

    return result


def extract_one_table(
        df: pd.DataFrame,
        n_stages: int,
        by_col: str,
        n_neighbors: int,
        threshold: int,
        rel_tol: float,
        abs_mean_level: float,
        abs_tol: float,
        metrics: list[str],
        circular_col: str,
        const_cols: list[str],
        min_stage_len: int,
) -> dict:
    """Extract stage features from one worksheet.

    :param df: Raw worksheet.
    :param n_stages: Number of stages.
    :param by_col: Column used to detect stage boundaries.
    :param n_neighbors: Look-ahead length.
    :param threshold: Minimum plateau length.
    :param rel_tol: Relative gap that ends a plateau.
    :param abs_mean_level: Level below which abs_tol is used.
    :param abs_tol: Absolute gap that ends a low plateau.
    :param metrics: Color columns to average.
    :param circular_col: Column averaged on the circle.
    :param const_cols: Camera columns copied from the first row.
    :param min_stage_len: Minimum divisor for an empty stage.
    :returns: Stage feature dictionary.
    :raise ValueError: a required column is missing
    """
    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]
    df         = df.rename(columns=COL_MAP)

    missing = [c for c in ["t"] + metrics if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    for col in const_cols:
        if col not in df.columns:
            df[col] = np.nan

    stage_info = analysis_stage(
        df, n_stages, by_col, n_neighbors, threshold, rel_tol, abs_mean_level, abs_tol,
    )
    result = extract_by_stage(
        df, n_stages, by_col, stage_info, metrics, circular_col, const_cols, min_stage_len,
    )

    return result


def build_dict(
        input_dir: str,
        n_stages: int,
        by_col: str,
        n_neighbors: int,
        threshold: int,
        rel_tol: float,
        abs_mean_level: float,
        abs_tol: float,
        metrics: list[str],
        circular_col: str,
        const_cols: list[str],
        min_stage_len: int,
        sheet_index: int,
) -> dict:
    """Read every Excel file in a directory and return its stage features.

    :param input_dir: Directory of ``.xls`` and ``.xlsx`` files.
    :param n_stages: Number of stages.
    :param by_col: Column used to detect stage boundaries.
    :param n_neighbors: Look-ahead length.
    :param threshold: Minimum plateau length.
    :param rel_tol: Relative gap that ends a plateau.
    :param abs_mean_level: Level below which abs_tol is used.
    :param abs_tol: Absolute gap that ends a low plateau.
    :param metrics: Color columns to average.
    :param circular_col: Column averaged on the circle.
    :param const_cols: Camera columns copied from the first row.
    :param min_stage_len: Minimum divisor for an empty stage.
    :param sheet_index: Worksheet index.
    :returns: ``{filename: feature dictionary}``.
    """
    input_dir = Path(input_dir)
    result = {}

    files = sorted(list(input_dir.glob("*.xlsx")) + list(input_dir.glob("*.xls")))
    if not files:
        print(f"no Excel files found: {input_dir.resolve()}")
        return result

    for file in files:
        try:
            xls = pd.ExcelFile(file)
        except Exception as e:
            print(f"cannot read {file}: {e}")
            continue

        sheets = xls.sheet_names
        if sheet_index < 0 or sheet_index >= len(sheets):
            print(f"sheet index {sheet_index} is outside {file.name}")
            continue

        sheet = sheets[sheet_index]
        try:
            df = cast(pd.DataFrame, pd.read_excel(xls, sheet_name=sheet))
            if df.empty:
                continue
            result[file.name] = extract_one_table(
                df, n_stages, by_col, n_neighbors, threshold, rel_tol,
                abs_mean_level, abs_tol, metrics, circular_col, const_cols, min_stage_len,
            )
            print(f"processed: {file.name} / {sheet}, rows={len(df)}")
        except Exception as e:
            print(f"failed: {file.name} / {sheet}: {e}")

    return result


def connect_data_and_ratio(
        data_dict: dict,
        id_ratio_dict_file: str,
        key_stage: int,
        metrics: list[str],
        const_cols: list[str],
) -> pd.DataFrame:
    """Join one stage of each file to its blue volume fraction.

    ``data_dict`` maps a filename to stage dictionaries plus camera columns.
    The mapping file maps that filename to ``ratio``.

    :param data_dict: Output of ``build_dict``.
    :param id_ratio_dict_file: ``.npy`` file storing ``{filename: ratio}``.
    :param key_stage: Stage written into the regression table.
    :param metrics: Color columns taken from the selected stage.
    :param const_cols: Camera columns taken from the file entry.
    :returns: One row per file.
    """
    data_ = data_dict.copy()
    dict_ = np.load(id_ratio_dict_file, allow_pickle=True).item()

    columns = {metric: [] for metric in metrics}
    for col in const_cols:
        columns[col] = []
    ratio_data = []

    for filename in data_.keys():
        stage_dict = data_[filename][key_stage]
        photo_dict = data_[filename]

        for metric in metrics:
            columns[metric].append(stage_dict[metric])
        for col in const_cols:
            columns[col].append(photo_dict[col])
        ratio_data.append(dict_[filename])

    columns["ratio"] = ratio_data
    return pd.DataFrame(columns)


def main():
    args = parse_args()
    metrics = split_csv(args.metrics)
    const_cols = split_csv(args.const_cols)

    os.makedirs(args.output_dir, exist_ok=True)
    output_data_file = os.path.join(args.output_dir, args.feature_filename)
    output_df_file   = os.path.join(args.output_dir, args.table_filename)

    data_dict = build_dict(
        input_dir      = args.input_dir,
        n_stages       = args.n_stages,
        by_col         = args.by_col,
        n_neighbors    = args.n_neighbors,
        threshold      = args.threshold,
        rel_tol        = args.rel_tol,
        abs_mean_level = args.abs_mean_level,
        abs_tol        = args.abs_tol,
        metrics        = metrics,
        circular_col   = args.circular_col,
        const_cols     = const_cols,
        min_stage_len  = args.min_stage_len,
        sheet_index    = args.sheet_index,
    )

    print("filenames:", list(data_dict.keys()))

    if data_dict:
        for filename, features in data_dict.items():
            print("\n\n\n")
            print(f"{filename}:\n")
            for stage, stage_features in features.items():
                print(f"\t{stage}:\n")
                print(f"\t\t{stage_features}")
                break

    np.save(output_data_file, data_dict)
    print(f"\nstage features saved to\n\t{output_data_file}")

    data_ratio_df = connect_data_and_ratio(
        data_dict, args.dict_file, args.key_stage, metrics, const_cols,
    )
    data_ratio_df.to_pickle(output_df_file)
    print(f"\nregression table saved to\n\t{output_df_file}")


if __name__ == "__main__":
    main()
