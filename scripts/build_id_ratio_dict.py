"""Build a filename-to-ratio map from the phyphox sample sheet."""

import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse the sample-sheet path and the output path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file",    required=False, type=str, default=str(ROOT / "data" / "2_settings" / "setting.csv"),
                        help="sample sheet csv")
    parser.add_argument("-o", "--output_dir",    required=False, type=str, default=str(ROOT / "data" / "2_settings"),
                        help="directory for the mapping file")
    parser.add_argument("-k", "--key",           required=False, type=str, default="planned_filename",
                        help="column used as the mapping key (phyphox filename)")
    parser.add_argument("-v", "--value",         required=False, type=str, default="true_blue_fraction",
                        help="column used as the mapping value (blue volume fraction)")
    parser.add_argument("--output_filename",     required=False, type=str, default="mapping.npy",
                        help="filename of the saved mapping")

    return parser.parse_args()


def main():
    args = parse_args()

    output_file = os.path.join(args.output_dir, args.output_filename)
    df = pd.read_csv(args.input_file)

    if df[args.key].duplicated().any():
        raise ValueError(f"duplicate values in column {args.key}")
    mapping = dict(zip(df[args.key], df[args.value]))

    for key, value in mapping.items():
        print(f"{key}\t: \t{value}\n")

    np.save(output_file, mapping)
    print(f"mapping saved to {output_file}")

if __name__ == "__main__":
    main()
