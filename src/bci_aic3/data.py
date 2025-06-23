# src/data.py

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from bci_aic3.paths import LABEL_MAPPING_PATH, RAW_DATA_DIR, TRAINING_STATS_PATH
from bci_aic3.util import (
    ensure_base_path,
    read_json_to_dict,
    save_training_stats,
    apply_normalization,
)


class BCIDataset(Dataset):
    def __init__(
        self,
        csv_file,
        base_path: Path | str,
        split: str,
        task_type: str,
        label_mapping: Optional[Dict[str, int]] = None,
        num_channels: int = 8,
    ):
        base_path = ensure_base_path(base_path)

        # Filter the main dataframe for the specific task (MI or SSVEP)
        self.metadata = pd.read_csv(os.path.join(base_path, csv_file))
        self.metadata = self.metadata[self.metadata["task"] == task_type]
        self.base_path = base_path
        self.task_type = task_type
        self.split = split
        self.label_mapping = label_mapping
        self.training_mean = None
        self.training_std = None

        # 9 seconds * 250 Hz = 2250 for MI
        # 7 seconds * 250 Hz = 1750 for SSVEP
        self.sequence_length = 2250 if task_type == "MI" else 1750

        num_trials = len(self.metadata)

        self.tensor_data = torch.empty(
            num_trials, num_channels, self.sequence_length, dtype=torch.float32
        )
        self.labels = torch.empty(num_trials, dtype=torch.long)

        # Precompute common part of the path
        base_task_split = self.base_path / task_type / self.split

        for i, (idx, row) in tqdm(
            enumerate(self.metadata.iterrows()), total=num_trials
        ):
            # Path to the EEG data file
            eeg_path = (
                base_task_split
                / row["subject_id"]
                / str(row["trial_session"])
                / "EEGdata.csv"
            )

            eeg_data = pd.read_csv(eeg_path)

            # Extract the correct trial segment
            trial_num = int(row["trial"])

            samples_per_trial = self.sequence_length
            start_idx = (trial_num - 1) * samples_per_trial
            end_idx = start_idx + samples_per_trial - 1

            # Select only the 8 EEG channels
            eeg_channels = ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]
            trial_data = eeg_data.loc[start_idx:end_idx, eeg_channels].values.T

            # uncomment the line below and comment the one above to include all 18 columns
            # trial_data = eeg_data.loc[start_idx:end_idx-1].values

            # Convert to tensor
            tensor_data = torch.tensor(trial_data, dtype=torch.float32)
            self.tensor_data[i] = tensor_data

            # Get label if it exists
            if "label" in row and self.label_mapping:
                label_str = row["label"]
                label_int = self.label_mapping[label_str]
                self.labels[i] = label_int

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        if self.split == "test":
            return self.tensor_data[idx]
        else:
            return self.tensor_data[idx], self.labels[idx]


def load_raw_data(
    base_path: str | Path,
    task_type: str,
    label_mapping: Optional[Dict[str, int]] = None,
    normalize: bool = False,
) -> Tuple[BCIDataset, BCIDataset, BCIDataset]:
    """
    Loads the train, val, test data for the given {task_type} inside the given {base_path}

    If no label_mapping is passed the data is loaded with the original labels.

    Returns:
        a tuple of BCIDataset in the order (train, val, test)
    """

    # Convert base_path to Path object if it's a string
    if isinstance(base_path, str):
        base_path = Path(base_path)

    train = BCIDataset(
        csv_file="train.csv",
        base_path=base_path,
        task_type=task_type,
        split="train",
        label_mapping=label_mapping,
    )

    val = BCIDataset(
        csv_file="validation.csv",
        base_path=base_path,
        task_type=task_type,
        split="validation",
        label_mapping=label_mapping,
    )
    test = BCIDataset(
        csv_file="test.csv",
        base_path=base_path,
        task_type=task_type,
        split="test",
        label_mapping=label_mapping,
    )

    return train, val, test


def load_processed_data(
    processed_data_dir: str | Path,
    task_type: str,
    normalize: bool = True,
) -> Tuple[TensorDataset, TensorDataset]:
    # Convert base_path to Path object if it's a string
    if isinstance(processed_data_dir, str):
        processed_data_dir = Path(processed_data_dir)

    data_path = processed_data_dir / task_type

    train_data = np.load(data_path / "train_data.npy")
    train_labels = np.load(data_path / "train_labels.npy")

    val_data = np.load(data_path / "validation_data.npy")
    val_labels = np.load(data_path / "validation_labels.npy")

    if normalize:
        training_means = train_data.mean((0, 2))
        training_stds = train_data.std((0, 2))

        save_training_stats(
            {"mean": training_means, "std": training_stds},
            TRAINING_STATS_PATH / f"{task_type.lower()}_stats.pt",
        )

        train_data = apply_normalization(train_data, training_means, training_stds)
        val_data = apply_normalization(val_data, training_means, training_stds)

    train_dataset = create_tensor_dataset(train_data, train_labels)
    val_dataset = create_tensor_dataset(val_data, val_labels)

    return train_dataset, val_dataset


def create_tensor_dataset(data: np.ndarray, labels: np.ndarray):
    tensor_data = torch.from_numpy(data).float()
    tesnor_labels = torch.from_numpy(labels).long()

    return TensorDataset(tensor_data, tesnor_labels)


def main():
    # This is what I do for reproducability
    label_mapping = read_json_to_dict(LABEL_MAPPING_PATH)

    # You can just use this to work with the data
    # label_mapping = {"Left": 0, "Right": 1, "Forward": 2, "Backward": 3}

    # Example of how to load the data using load_data
    train_mi, val_mi, test_mi = load_raw_data(
        base_path=RAW_DATA_DIR, task_type="MI", label_mapping=label_mapping
    )

    train_ssvep, val_ssvep, test_ssvep = load_raw_data(
        base_path=RAW_DATA_DIR, task_type="SSVEP", label_mapping=label_mapping
    )

    print(len(train_mi))
    print(len(test_ssvep))


if __name__ == "__main__":
    main()
