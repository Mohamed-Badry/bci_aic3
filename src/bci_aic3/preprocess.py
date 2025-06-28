import mne
import numpy as np
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import DataLoader
from torcheeg import transforms
from torcheeg.transforms import Lambda, BaseTransform

from bci_aic3.config import ProcessingConfig
from bci_aic3.data import BCIDataset
from bci_aic3.paths import (
    PROCESSED_DATA_DIR,
    TRAINING_STATS_PATH,
)

# Copied from notebook as is probably needs some tweaks to work as a script and remove the side effects


class MNENotchFilter:
    """A callable transform class to apply a notch filter using MNE."""

    def __init__(self, config):
        self.sfreq = config.sfreq
        self.notch_freq = config.notch_freq

    def __call__(self, **kwargs):
        """Apply the notch filter to the EEG data."""
        eeg = kwargs["eeg"]

        # Convert to numpy if torch tensor
        if isinstance(eeg, torch.Tensor):
            device = eeg.device
            dtype = eeg.dtype
            eeg_np = eeg.detach().cpu().numpy()
            was_torch = True
        else:
            eeg_np = eeg
            was_torch = False

        eeg_np = eeg_np.astype(float)
        filtered = mne.filter.notch_filter(
            eeg_np,
            Fs=self.sfreq,
            freqs=self.notch_freq,
            method="iir",
            verbose=False,
        )

        # Convert back to torch if input was torch
        if was_torch:
            kwargs["eeg"] = torch.from_numpy(filtered).to(device=device, dtype=dtype)
        else:
            kwargs["eeg"] = filtered

        return kwargs


class BandPassFilter:
    """Simple bandpass filter for EEG signals."""

    def __init__(self, config):
        self.sfreq = config.sfreq
        self.lfreq = config.bandpass_low
        self.hfreq = config.bandpass_high
        self.order = config.filter_order

        # Pre-compute filter coefficients
        nyquist = self.sfreq / 2
        low = self.lfreq / nyquist
        high = self.hfreq / nyquist
        self.b, self.a = butter(self.order, [low, high], btype="bandpass")  # type: ignore

    def __call__(self, **kwargs):
        """Apply bandpass filter to EEG data."""
        eeg = kwargs["eeg"]

        # Convert to torch if numpy
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg).float()
            was_numpy = True
        else:
            was_numpy = False

        device = eeg.device
        dtype = eeg.dtype

        # Convert to numpy and filter
        eeg_np = eeg.detach().cpu().numpy()

        if eeg_np.ndim == 2:
            filtered = np.array([filtfilt(self.b, self.a, ch) for ch in eeg_np])
        else:  # 3D
            filtered = np.array(
                [[filtfilt(self.b, self.a, ch) for ch in batch] for batch in eeg_np]
            )

        kwargs["eeg"] = torch.from_numpy(filtered).to(device=device, dtype=dtype)
        return kwargs


class CustomCrop:
    """A callable transform class to crop the EEG signal in time."""

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __call__(self, **kwargs):
        """Apply temporal cropping to the EEG data."""
        eeg = kwargs["eeg"]

        if isinstance(eeg, torch.Tensor):
            kwargs["eeg"] = eeg[:, self.start : self.end]
        else:
            kwargs["eeg"] = eeg[:, self.start : self.end]

        return kwargs


class FixedMeanStdNormalize(BaseTransform):
    """
    Normalize EEG data using precomputed mean and std per channel.

    Args:
        mean: Tensor of shape (n_channels,) with mean per channel.
        std: Tensor of shape (n_channels,) with std per channel.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, **kwargs):
        """
        Apply normalization to a single EEG sample.

        Args:
            data: Tensor of shape (n_channels, n_timepoints).

        Returns:
            Normalized tensor of the same shape.
        """
        eeg = kwargs["eeg"]

        if isinstance(eeg, torch.Tensor):
            kwargs["eeg"] = (eeg - self.mean[:, None]) / self.std[:, None]
        else:
            kwargs["eeg"] = (eeg - self.mean[:, None]) / self.std[:, None]
        return kwargs


class ChannelWiseNormalizer:
    """A callable transform class for channel-wise z-score normalization."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    def __call__(self, **kwargs):
        """Apply normalization to the EEG data."""
        eeg = kwargs["eeg"]

        if isinstance(eeg, torch.Tensor):
            mean_tensor = torch.from_numpy(self.mean).to(eeg.device, eeg.dtype)
            std_tensor = torch.from_numpy(self.std).to(eeg.device, eeg.dtype)
            kwargs["eeg"] = (eeg - mean_tensor) / std_tensor
        else:
            kwargs["eeg"] = (eeg - self.mean) / self.std

        return kwargs


def preprocessing_pipeline(
    train_dataset: BCIDataset,
    validation_dataset: BCIDataset,
    task_type: str,
    processing_config: ProcessingConfig,
):
    """
    The main pipeline to preprocess data using a torcheeg-based approach.

    This pipeline:
    1. Defines a series of transforms (Notch, Bandpass, Crop, Normalize).
    2. Applies initial transforms to the training data to calculate normalization stats.
    3. Creates a full pipeline with the calculated stats.
    4. Applies the full pipeline to both training and validation data.
    5. Saves the processed data to disk.
    """
    # Load all data into memory from the datasets
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    train_data, train_labels = next(iter(train_loader))
    train_data, train_labels = train_data.numpy(), train_labels.numpy()

    val_loader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset), shuffle=False
    )
    val_data, val_labels = next(iter(val_loader))
    val_data, val_labels = val_data.numpy(), val_labels.numpy()

    # --- Step 1: Define initial transforms (before normalization) ---
    start_idx = int(processing_config.tmin * processing_config.sfreq)
    end_idx = int(processing_config.tmax * processing_config.sfreq)

    initial_transforms = None
    if task_type.upper() == "MI":
        initial_transforms = transforms.Compose(
            [
                MNENotchFilter(config=processing_config),
                BandPassFilter(config=processing_config),
                CustomCrop(start=start_idx, end=end_idx),
            ]
        )
    elif task_type.upper() == "SSVEP":
        initial_transforms = transforms.Compose(
            [
                MNENotchFilter(config=processing_config),
                BandPassFilter(config=processing_config),
                CustomCrop(start=start_idx, end=end_idx),
            ]
        )
    else:
        raise (
            ValueError(
                f"Invalid task_type: {task_type}.\nValid task_type (MI) or (SSVEP)"
            )
        )

    # --- Step 2: Apply initial transforms to training data to get stats ---
    print("Applying initial transforms to calculate normalization statistics...")
    train_data_for_stats = np.array(
        [initial_transforms(eeg=epoch) for epoch in train_data]
    )

    # Extract EEG data from the array of dictionaries
    print(train_data_for_stats)
    train_eeg_data = np.array([result["eeg"] for result in train_data_for_stats])

    print(train_eeg_data)
    # Calculate channel-wise normalization statistics from the transformed training data
    mean = np.mean(train_eeg_data, axis=(0, 2), keepdims=True)
    std = np.std(train_eeg_data, axis=(0, 2), keepdims=True)
    std[std == 0] = 1e-6  # Add a small epsilon to std to avoid division by zero

    print(f"Calculated Stats --- Mean shape: {mean.shape}, Std shape: {std.shape}")

    # --- Step 3: Create the full, extensible processing pipeline ---
    shift_samples = int(processing_config.percent_shifted * processing_config.sfreq)

    training_transforms = transforms.Compose(
        [
            *initial_transforms.transforms,  # Unpack the initial transforms
            # Lambda(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x),
            # transforms.MeanStdNormalize(axis=1),
            # transforms.ToTensor(),
            FixedMeanStdNormalize(mean, std),
            transforms.RandomNoise(p=processing_config.p_noise, mean=0, std=0.1),
            transforms.RandomShift(
                p=processing_config.p_shift,
                shift_min=-shift_samples,
                shift_max=shift_samples,
            ),  # type: ignore
            Lambda(lambda x: x.squeeze(0)),
        ]
    )

    validation_transforms = transforms.Compose(
        [
            *initial_transforms.transforms,  # Unpack the initial transforms
            # Lambda(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x),
            FixedMeanStdNormalize(mean, std),
            # transforms.MeanStdNormalize(axis=1),
            # transforms.ToTensor(),
            Lambda(lambda x: x.squeeze(0)),
        ]
    )

    print(
        f"\nPipeline created with {len(training_transforms.transforms)} steps: "
        f"{[t.__class__.__name__ for t in training_transforms.transforms]}"
    )

    # --- Step 4: Apply the final pipeline to all datasets ---
    print("\nTransforming training data with the full pipeline...")
    train_data_processed = np.array(
        [training_transforms(eeg=epoch)["eeg"] for epoch in train_data]
    )
    print(f"Processed training data shape: {train_data_processed.shape}")
    print("Done.")

    norm = (train_data_processed - mean[:None]) / std[:None]
    print(
        f"\nNormalized training data mean and std: \n{np.mean(norm, axis=(0, 3))}, \n{np.std(norm, axis=(0, 3))}, \n {np.std(norm, axis=(0, 2))}, \n{np.std(norm, axis=(0, 1, 3))}"
    )

    print("Transforming validation data with the full pipeline...")
    val_data_processed = np.array(
        [validation_transforms(eeg=epoch)["eeg"] for epoch in val_data]
    )
    print(f"Processed validation data shape: {val_data_processed.shape}")
    print("Done.")

    # --- Step 5: Save the processed data ---
    output_dir = PROCESSED_DATA_DIR / task_type.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_data.npy", train_data_processed)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "validation_data.npy", val_data_processed)
    np.save(output_dir / "validation_labels.npy", val_labels)
    print(f"\n✅ Processed data successfully saved to '{output_dir}'")


def main():
    print("This is Hell.")


if __name__ == "__main__":
    main()
