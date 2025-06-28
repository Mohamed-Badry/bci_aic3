import argparse
import glob
import pandas as pd
from pathlib import Path


from bci_aic3.inference import make_inference
from bci_aic3.paths import (
    BEST_MODELS_CHECKPOINT,
    SUBMISSIONS_DIR,
    RAW_DATA_DIR,
)
from bci_aic3.train import BCILightningModule


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
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--mi",
        help="Path to the MI model checkpoint (.ckpt) file.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--ssvep",
        help="Path to the SSVEP model checkpoint (.ckpt) file.",
        default=None,
        type=Path,
    )
    args = parser.parse_args()

    # Find single .ckpt file in each directory if user didn't provide it
    if args.mi is None:
        mi_ckpt_files = glob.glob(str(BEST_MODELS_CHECKPOINT / "MI" / "*.ckpt"))
        if len(mi_ckpt_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 .ckpt file in {BEST_MODELS_CHECKPOINT / 'MI'} directory, found {len(mi_ckpt_files)}"
            )
        mi_model_path = Path(mi_ckpt_files[0])
    else:
        mi_model_path = Path(args.mi) if isinstance(args.mi, str) else args.mi

    if args.ssvep is None:
        ssvep_ckpt_files = glob.glob(str(BEST_MODELS_CHECKPOINT / "SSVEP" / "*.ckpt"))
        if len(ssvep_ckpt_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 .ckpt file in {BEST_MODELS_CHECKPOINT / 'SSVEP'} directory, found {len(ssvep_ckpt_files)}"
            )
        ssvep_model_path = Path(ssvep_ckpt_files[0])
    else:
        ssvep_model_path = (
            Path(args.ssvep) if isinstance(args.ssvep, str) else args.ssvep
        )

    mi_model = BCILightningModule.load_from_checkpoint(mi_model_path)
    ssvep_model = BCILightningModule.load_from_checkpoint(ssvep_model_path)

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

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    save_labels(
        csv_file_path=args.base_path / args.csv_file,
        mi_preds=mi_preds,
        ssvep_preds=ssvep_preds,
        output_file_path=SUBMISSIONS_DIR / args.output_file,
    )


if __name__ == "__main__":
    main()
