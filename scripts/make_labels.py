import argparse
from pathlib import Path

from bci_aic3.inference import make_inference
from bci_aic3.paths import SUBMISSIONS_DIR, RAW_DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default=RAW_DATA_DIR, help="Path to data directory", type=Path
    )
    parser.add_argument(
        "--output_path", default=SUBMISSIONS_DIR, help="Output directory", type=Path
    )
    args = parser.parse_args()
    print(args)

    print(Path(args.data_path))
    print(Path(args.output_path))


if __name__ == "__main__":
    main()
