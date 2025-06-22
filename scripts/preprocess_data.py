import argparse

from bci_aic3.paths import (
    MI_CONFIG_PATH,
    SSVEP_CONFIG_PATH,
    RAW_DATA_DIR,
    LABEL_MAPPING_PATH,
)
from bci_aic3.data import BCIDataset
from bci_aic3.util import read_json_to_dict
from bci_aic3.config import load_processing_config
from bci_aic3.preprocess import preprocessing_pipeline


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

    train = BCIDataset(
        "train.csv",
        base_path=RAW_DATA_DIR,
        task_type=task_type,
        split="train",
        label_mapping=label_mapping,
    )

    val = BCIDataset(
        "validation.csv",
        base_path=RAW_DATA_DIR,
        task_type=task_type,
        split="validation",
        label_mapping=label_mapping,
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
