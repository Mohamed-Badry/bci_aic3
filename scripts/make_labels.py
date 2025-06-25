import argparse
import pandas as pd
from pathlib import Path


from bci_aic3.inference import make_inference
from bci_aic3.paths import (
    SUBMISSIONS_DIR,
    RAW_DATA_DIR,
)
from bci_aic3.util import load_model


def save_labels(csv_file_path, mi_preds, ssvep_preds, output_file_path):
    df = pd.read_csv(csv_file_path)

    df.loc[df["task"] == "SSVEP", "labels"] = ssvep_preds
    df.loc[df["task"] == "MI", "labels"] = mi_preds

    df[["id", "labels"]].to_csv(output_file_path, index=False)
    print(f"Labels successfully saved to the file {output_file_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        default=RAW_DATA_DIR,
        help="Path to data directory",
        type=Path,
    )
    parser.add_argument(
        "--csv_file",
        default="test.csv",
        help="Name of the metadata csv file to perform inference and create labels for.",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="Output csv file name (submission.csv) for example.",
        type=Path,
    )
    parser.add_argument(
        "--mi_model_dir",
        help="Path to the MI model directory that has the weights.pt file.",
        type=Path,
    )
    parser.add_argument(
        "--ssvep_model_dir",
        help="Path to the SSVEP model directorythat has the weights.pt file.",
        type=Path,
    )
    args = parser.parse_args()

    mi_model = load_model(model_path=Path(args.mi_model_dir) / "weights.pt")
    ssvep_model = load_model(model_path=Path(args.ssvep_model_dir) / "weights.pt")

    mi_preds = make_inference(
        mi_model,
        csv_file=args.csv_file,
        base_path=args.base_path,
        task_type="MI",
        reverse_mapping=True,
    )

    ssvep_preds = make_inference(
        ssvep_model,
        csv_file=args.csv_file,
        base_path=args.base_path,
        task_type="SSVEP",
        reverse_mapping=True,
    )

    save_labels(
        csv_file_path=args.base_path / args.csv_file,
        mi_preds=mi_preds,
        ssvep_preds=ssvep_preds,
        output_file_path=SUBMISSIONS_DIR / args.output_file,
    )


if __name__ == "__main__":
    main()
