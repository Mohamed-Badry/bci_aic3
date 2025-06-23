import argparse

from bci_aic3.config import load_processing_config
from bci_aic3.data import load_raw_data
from bci_aic3.paths import (
    LABEL_MAPPING_PATH,
    MI_CONFIG_PATH,
    RAW_DATA_DIR,
    SSVEP_CONFIG_PATH,
)
from bci_aic3.preprocess import preprocessing_pipeline
from bci_aic3.util import read_json_to_dict

"""
Preprocess raw EEG data for MI or SSVEP tasks and save processed datasets.
This script loads raw EEG data, applies preprocessing steps according to the
specified task type (MI or SSVEP), and saves the resulting processed data and
labels as NumPy files in the PROCESSED_DATA_DIR directory. The output consists
of four files:
    - train_data.npy: Preprocessed training data
    - train_labels.npy: Corresponding labels for training data
    - validation_data.npy: Preprocessed validation data
    - validation_labels.npy: Corresponding labels for validation data
Usage:
    python preprocess_data.py --task_type <MI|SSVEP>
Arguments:
    --task_type: The type of BCI task to preprocess ("MI" for Motor Imagery or "SSVEP" for Steady-State Visual Evoked Potential).
The script expects configuration files and raw data to be available at the
locations specified in bci_aic3.paths. Preprocessing steps are defined in the
configuration files and applied via the preprocessing_pipeline function.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        required=True,
        help="Task type (MI or SSVEP).",
    )
    args = parser.parse_args()
    task_type = args.task_type.upper()

    config_path = None
    if task_type == "MI":
        config_path = MI_CONFIG_PATH
    elif task_type == "SSVEP":
        config_path = SSVEP_CONFIG_PATH
    else:
        raise (
            ValueError(
                f"Invalid task_type: {task_type}.\nValid task_type (MI) or (SSVEP)"
            )
        )

    processing_config = load_processing_config(config_path)
    label_mapping = read_json_to_dict(LABEL_MAPPING_PATH)

    train, val, _ = load_raw_data(
        base_path=RAW_DATA_DIR,
        task_type=task_type,
        label_mapping=label_mapping,
        normalize=False,
    )

    preprocessing_pipeline(
        train,
        task_type=task_type,
        split="train",
        processing_config=processing_config,
    )

    preprocessing_pipeline(
        val,
        task_type=task_type,
        split="validation",
        processing_config=processing_config,
    )


if __name__ == "__main__":
    main()
