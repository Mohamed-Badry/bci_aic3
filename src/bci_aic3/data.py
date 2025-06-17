# data.py
import os
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset

from bci_aic3.util import read_json_to_dict


BASE_PATH = "/kaggle/input/mtcaic3"
# base_path = "/data/raw/mtcaic3"
LABEL_MAPPING_JSON_PATH = "../../configs/label_mapping.json"


class BCIDataset(Dataset):
    def __init__(self, csv_file, base_path, task_type="MI", label_mapping=None):
        # Filter the main dataframe for the specific task (MI or SSVEP)
        self.metadata = pd.read_csv(os.path.join(base_path, csv_file))
        self.metadata = self.metadata[self.metadata["task"] == task_type]
        self.base_path = base_path
        self.task_type = task_type
        self.label_mapping = label_mapping

        # 9 seconds * 250 Hz = 2250 for MI
        # 7 seconds * 250 Hz = 1750 for SSVEP
        self.sequence_length = 2250 if task_type == "MI" else 1750

        num_trials = len(self.metadata)

        self.tensor_data = torch.empty(
            num_trials, self.sequence_length, 8, dtype=torch.float32
        )
        self.labels = torch.empty(num_trials, dtype=torch.long)

        for i, (idx, row) in enumerate(self.metadata.iterrows()):
            # Determine dataset split (train/validation/test)
            id_num = row["id"]
            if id_num <= 4800:
                dataset_split = "train"
            elif id_num <= 4900:
                dataset_split = "validation"
            else:
                dataset_split = "test"

            # Path to the EEG data file
            eeg_path = os.path.join(
                self.base_path,
                row["task"],
                dataset_split,
                row["subject_id"],
                str(row["trial_session"]),
                "EEGdata.csv",
            )

            eeg_data = pd.read_csv(eeg_path)

            # Extract the correct trial segment
            trial_num = int(row["trial"])

            samples_per_trial = self.sequence_length
            start_idx = (trial_num - 1) * samples_per_trial
            end_idx = start_idx + samples_per_trial - 1

            # Select only the 8 EEG channels
            eeg_channels = ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]
            trial_data = eeg_data.loc[start_idx:end_idx, eeg_channels].values

            # uncomment the line below and comment the one above to include all 18 columns
            # trial_data = eeg_data.loc[start_idx:end_idx-1].values

            # Preprocess the data (see next section)
            processed_data = self.preprocess(trial_data)

            # Convert to tensor
            tensor_data = torch.tensor(processed_data, dtype=torch.float32)
            self.tensor_data[i] = tensor_data

            # Get label if it exists
            if "label" in row and self.label_mapping:
                label_str = row["label"]
                label_int = self.label_mapping[label_str]
                self.labels[i] = label_int

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        if self.labels[idx] is not None:
            return self.tensor_data[idx], self.labels[idx]
        else:
            return self.tensor_data[idx]

    def preprocess(self, eeg_data):
        # Apply preprocessing steps here (filtering, normalization, etc.)
        # This will be different for MI and SSVEP
        # ...
        return eeg_data


def load_data(
    base_path, task_type, label_mapping=None
) -> Tuple[BCIDataset, BCIDataset, BCIDataset]:
    """
    Loads the train, val, test data for the given {task_type} inside the given {base_path}

    Returns:
        a tuple of BCIDataset in the order (train, val, test)
    """

    train = BCIDataset(
        csv_file="train.csv",
        base_path=base_path,
        task_type=task_type,
        label_mapping=label_mapping,
    )
    val = BCIDataset(
        csv_file="validation.csv",
        base_path=base_path,
        task_type=task_type,
        label_mapping=label_mapping,
    )
    test = BCIDataset(
        csv_file="test.csv",
        base_path=base_path,
        task_type=task_type,
        label_mapping=label_mapping,
    )

    return train, val, test


def main():
    label_mapping = read_json_to_dict(LABEL_MAPPING_JSON_PATH)

    # Example of how to load the data using load_data
    train_mi, val_mi, test_mi = load_data(
        base_path=BASE_PATH, task_type="MI", label_mapping=label_mapping
    )

    train_ssvep, val_ssvep, test_ssvep = load_data(
        base_path=BASE_PATH, task_type="SSVEP", label_mapping=label_mapping
    )


if __name__ == "__main__":
    main()
