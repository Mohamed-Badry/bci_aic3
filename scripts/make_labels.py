import argparse

from bci_aic3.inference import make_inference
from bci_aic3.paths import SUBMISSIONS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to data directory")
    parser.add_argument(
        "--output_path", default=SUBMISSIONS_DIR, help="Output directory"
    )
    args = parser.parse_args()


if __name__ == "__main__":
    main()
