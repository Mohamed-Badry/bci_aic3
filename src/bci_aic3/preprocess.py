# src/preprocess.py

from pathlib import Path
from typing import Optional, Tuple

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bci_aic3.config import ProcessingConfig, load_processing_config
from bci_aic3.data import BCIDataset
from bci_aic3.paths import (
    LABEL_MAPPING_PATH,
    MI_CONFIG_PATH,
    PROCESSED_DATA_DIR,
    SSVEP_CONFIG_PATH,
    TRAINING_STATS_PATH,
)
from bci_aic3.util import apply_normalization, load_training_stats, read_json_to_dict


def apply_notch_filter(data: np.ndarray, sfreq: float, notch_freq: float) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.

    Args:
        data: EEG data, shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        notch_freq: Frequency to notch out (e.g., 50Hz or 60Hz)

    Returns:
        Notch filtered data
    """
    return mne.filter.notch_filter(data, sfreq, notch_freq, method="fft", verbose=False)


def apply_bandpass_filter(
    data: np.ndarray, sfreq: float, l_freq: float, h_freq: float
) -> np.ndarray:
    """
    Apply bandpass filter to focus on frequencies of interest.

    Args:
        data: EEG data, shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        l_freq: Low cutoff frequency
        h_freq: High cutoff frequency

    Returns:
        Bandpass filtered data
    """
    return mne.filter.filter_data(
        data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
    )


def apply_baseline_correction(
    data: np.ndarray, sfreq: float, baseline: Tuple[float, float]
) -> np.ndarray:
    """
    Apply baseline correction to remove DC offset and slow drifts.

    Args:
        data: EEG data, shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        baseline: Baseline period in seconds (start, end)

    Returns:
        Baseline corrected data
    """
    baseline_start_idx = int(baseline[0] * sfreq)
    baseline_end_idx = int(baseline[1] * sfreq)

    # Calculate baseline mean for each epoch and channel
    baseline_mean = np.mean(
        data[:, :, baseline_start_idx:baseline_end_idx], axis=2, keepdims=True
    )

    # Subtract baseline from entire epoch
    return data - baseline_mean


def crop_epochs(
    data: np.ndarray, sfreq: float, tmin: float, tmax: float, original_tmin: float = 0.0
) -> np.ndarray:
    """
    Crop epochs to focus on specific time window.

    Args:
        data: EEG data, shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        tmin: Start time for cropping
        tmax: End time for cropping
        original_tmin: Original start time of the data

    Returns:
        Cropped data
    """
    start_idx = int((tmin - original_tmin) * sfreq)
    end_idx = int((tmax - original_tmin) * sfreq)

    return data[:, :, start_idx:end_idx]


def apply_all_preprocessing_steps(
    data: np.ndarray, settings: ProcessingConfig
) -> np.ndarray:
    """
    Complete EEG preprocessing pipeline.

    Args:
        data: Raw EEG data, shape (n_epochs, n_channels, n_times)
        settings: ProcessingSettings object with all parameters

    Returns:
        Fully preprocessed EEG data
    """
    processed_data = data.astype(float) * settings.scaling_factor

    # Step 1: Apply notch filter to remove power line interference
    processed_data = apply_notch_filter(
        processed_data, settings.sfreq, settings.notch_freq
    )

    # Step 2: Apply bandpass filter to focus on relevant frequencies
    processed_data = apply_bandpass_filter(
        processed_data, settings.sfreq, settings.lfreq, settings.hfreq
    )

    # Step 3: Apply baseline correction
    processed_data = apply_baseline_correction(
        processed_data, settings.sfreq, settings.baseline
    )

    # Step 4: Crop to time window of interest
    processed_data = crop_epochs(
        processed_data, settings.sfreq, settings.tmin, settings.tmax
    )

    # Step 5: Scale back
    processed_data = processed_data / settings.scaling_factor

    return processed_data.astype(np.float32)


def preprocessing_pipeline(
    dataset: BCIDataset, task_type: str, split: str, processing_config: ProcessingConfig
):
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data_batch, labels = next(iter(data_loader))

    data = data_batch.numpy()
    labels = labels.numpy()

    processed_data = apply_all_preprocessing_steps(
        data=data, settings=processing_config
    )

    processed_data_path = PROCESSED_DATA_DIR / task_type.upper() / f"{split}_data.npy"
    processed_labels_path = (
        PROCESSED_DATA_DIR / task_type.upper() / f"{split}_labels.npy"
    )

    np.save(processed_data_path, processed_data)
    print(f"Processed data successfully saved at: {processed_data_path}")

    np.save(processed_labels_path, labels)
    print(f"Processed labels successfully saved at: {processed_labels_path}")


def load_and_preprocess_for_inference(
    csv_file: str, base_path: str | Path, task_type: str
) -> TensorDataset:
    test = BCIDataset(
        csv_file=csv_file,
        base_path=base_path,
        task_type=task_type,
        split="test",
        label_mapping=read_json_to_dict(LABEL_MAPPING_PATH),
    )

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

    data_loader = DataLoader(test, batch_size=len(test), shuffle=False)
    data_batch = next(iter(data_loader))

    data = data_batch.numpy()

    processed_data = apply_all_preprocessing_steps(
        data=data, settings=processing_config
    )

    training_stats = load_training_stats(
        TRAINING_STATS_PATH / f"{task_type.lower()}_stats.pt"
    )
    normalized_test_data = apply_normalization(
        processed_data, training_stats["mean"], training_stats["std"]
    )

    test_tensor = torch.from_numpy(normalized_test_data).float()

    return TensorDataset(test_tensor)


# TODO: Might use this for advanced preprocessing and artifact removal.
def create_mne_epochs(
    data: np.ndarray, ch_names: list, sfreq: float, labels: Optional[np.ndarray] = None
) -> mne.EpochsArray:
    """
    Create MNE EpochsArray object for advanced preprocessing.

    Args:
        data: EEG data, shape (n_epochs, n_channels, n_times)
        ch_names: List of channel names
        sfreq: Sampling frequency
        labels: Optional labels for events

    Returns:
        MNE EpochsArray object
    """
    # Create MNE info object
    # ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names, sfreq, "eeg")

    # Create epochs object
    epochs = mne.EpochsArray(
        data, info, events=None, event_id=None, tmin=0.0, verbose=False
    )

    # Add event information if labels provided
    if labels is not None:
        epochs.event_id = dict((str(lbl), int(lbl)) for lbl in np.unique(labels))
        epochs.events = np.column_stack(
            (np.arange(len(labels)), np.zeros(len(labels), int), labels)
        )

    return epochs


def main():
    processing_settings = load_processing_config(MI_CONFIG_PATH)
    print(processing_settings)


if __name__ == "__main__":
    main()
