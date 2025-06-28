from typing import Tuple, Union

import joblib
import mne
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline
from torch.utils.data import DataLoader, Dataset

from bci_aic3.config import ProcessingConfig
from bci_aic3.paths import (
    PROCESSED_DATA_DIR,
    TRAINING_STATS_PATH,
)

# Copied from notebook as is probably needs some tweaks to work as a script and remove the side effects


class MNENotchFilter(BaseEstimator, TransformerMixin):
    """Notch filter using MNE for EEG data."""

    def __init__(self, sfreq: float = 250.0, notch_freq: Union[float, list] = 50.0):
        self.sfreq = sfreq
        self.notch_freq = notch_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Apply notch filter to EEG data.

        Args:
            X: EEG data of shape (n_samples, n_channels, n_timepoints)

        Returns:
            Filtered EEG data of same shape
        """
        X_filtered = np.zeros_like(X)

        for i, epoch in enumerate(X):
            filtered = mne.filter.notch_filter(
                epoch.astype(float),
                Fs=self.sfreq,
                freqs=self.notch_freq,
                method="iir",
                verbose=False,
            )
            X_filtered[i] = filtered

        return X_filtered


class BandPassFilter(BaseEstimator, TransformerMixin):
    """Butterworth bandpass filter for EEG signals."""

    def __init__(
        self,
        sfreq: float = 250.0,
        low_freq: float = 1.0,
        high_freq: float = 40.0,
        order: int = 4,
    ):
        self.sfreq = sfreq
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self._filter_coeffs = None

    def fit(self, X, y=None):
        # Pre-compute filter coefficients
        nyquist = self.sfreq / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        self._filter_coeffs = butter(self.order, [low, high], btype="bandpass")
        return self

    def transform(self, X):
        """Apply bandpass filter to EEG data.

        Args:
            X: EEG data of shape (n_samples, n_channels, n_timepoints)

        Returns:
            Filtered EEG data of same shape
        """
        if self._filter_coeffs is None:
            raise ValueError("Filter not fitted. Call fit() first.")

        b, a = self._filter_coeffs  # type: ignore
        X_filtered = np.zeros_like(X)

        for i, epoch in enumerate(X):
            filtered = np.array([filtfilt(b, a, ch) for ch in epoch])
            X_filtered[i] = filtered

        return X_filtered


class TemporalCrop(BaseEstimator, TransformerMixin):
    """Crop EEG signal in time dimension."""

    def __init__(self, tmin: float = 0.0, tmax: float = 4.0, sfreq: float = 250.0):
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self._start_idx = None
        self._end_idx = None

    def fit(self, X, y=None):
        self._start_idx = int(self.tmin * self.sfreq)
        self._end_idx = int(self.tmax * self.sfreq)
        return self

    def transform(self, X):
        """Crop EEG data in time.

        Args:
            X: EEG data of shape (n_samples, n_channels, n_timepoints)

        Returns:
            Cropped EEG data of shape (n_samples, n_channels, cropped_timepoints)
        """
        return X[:, :, self._start_idx : self._end_idx]


class StatisticalArtifactRemoval(BaseEstimator, TransformerMixin):
    """Statistical artifact removal for ADC data without voltage conversion."""

    def __init__(self, z_threshold: float = 3.0, method: str = "iqr"):
        """
        Args:
            z_threshold: Z-score threshold for outlier detection
            method: 'zscore', 'iqr', or 'percentile'
        """
        self.z_threshold = z_threshold
        self.method = method
        self._threshold = None
        self.clean_indices_ = None

    def fit(self, X, y=None):
        if self.method == "zscore":
            # Use standard deviation based threshold
            std_vals = np.std(X, axis=2)
            self._threshold = np.mean(std_vals) + self.z_threshold * np.std(std_vals)
        elif self.method == "iqr":
            # Use interquartile range
            peak_to_peak = np.max(X, axis=2) - np.min(X, axis=2)
            q75, q25 = np.percentile(peak_to_peak, [75, 25])
            iqr = q75 - q25
            self._threshold = q75 + 1.5 * iqr
        elif self.method == "percentile":
            # Use percentile threshold
            peak_to_peak = np.max(X, axis=2) - np.min(X, axis=2)
            self._threshold = np.percentile(peak_to_peak, 95)

        return self

    def transform(self, X):
        """Remove artifacts based on statistical criteria."""
        if self._threshold is None:
            raise ValueError("Transformer not fitted. Call fit() first.")

        # Calculate rejection criteria per epoch
        if self.method == "zscore":
            epoch_metric = np.std(X, axis=2)
        else:
            epoch_metric = np.max(X, axis=2) - np.min(X, axis=2)

        # Find clean epochs (all channels below threshold)
        clean_mask = np.all(epoch_metric < self._threshold, axis=1)
        self.clean_indices_ = np.where(clean_mask)[0]

        if not np.any(clean_mask):
            print(f"Warning: All epochs rejected with {self.method} method")
            print("Returning original data")
            self.clean_indices_ = np.arange(len(X))
            return X

        return X[clean_mask]


class ChannelWiseNormalizer(BaseEstimator, TransformerMixin):
    """Channel-wise normalization (z-score) for EEG data."""

    def __init__(self, axis: Tuple[int, ...] = (0, 2)):
        """
        Args:
            axis: Axes over which to compute mean and std for normalization.
                  Default (0, 2) normalizes across samples and time for each channel.
        """
        self.axis = axis
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """Compute channel-wise statistics from training data.

        Args:
            X: EEG data of shape (n_samples, n_channels, n_timepoints)
        """
        self.mean_ = np.mean(X, axis=self.axis, keepdims=True)
        self.std_ = np.std(X, axis=self.axis, keepdims=True)

        # Avoid division by zero
        self.std_[self.std_ == 0] = 1e-6

        return self

    def transform(self, X):
        """Apply channel-wise normalization.

        Args:
            X: EEG data of shape (n_samples, n_channels, n_timepoints)

        Returns:
            Normalized EEG data of same shape
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        return (X - self.mean_) / self.std_


class EEGReshaper(BaseEstimator, TransformerMixin):
    """Reshape EEG data for different model requirements."""

    def __init__(self, target_shape: str = "flatten"):
        """
        Args:
            target_shape: 'flatten' to reshape to (n_samples, n_features)
                         'keep' to maintain original shape
        """
        self.target_shape = target_shape
        self.original_shape_ = None

    def fit(self, X, y=None):
        self.original_shape_ = X.shape[1:]  # Store shape without sample dimension
        return self

    def transform(self, X):
        """Reshape EEG data."""
        if self.target_shape == "flatten":
            return X.reshape(X.shape[0], -1)
        return X

    def inverse_transform(self, X):
        """Restore original shape."""
        if self.target_shape == "flatten" and self.original_shape_ is not None:
            return X.reshape(X.shape[0], *self.original_shape_)
        return X


def unsqueeze_for_eeg(X):
    """
    Unsqueezes the input data to add a new dimension for the channel axis.
    """
    return np.expand_dims(X, axis=1)


def create_eeg_pipeline(
    task_type: str, processing_config: ProcessingConfig, test: bool = False
):
    """
    Create a scikit-learn pipeline for EEG preprocessing.

    Args:
        task_type: 'MI' or 'SSVEP'
        processing_config: Configuration object with processing parameters
        test: when set to true artifact removal transform is excluded

    Returns:
        sklearn.pipeline.Pipeline
    """

    # Define pipeline steps
    steps = [
        (
            "notch_filter",
            MNENotchFilter(
                sfreq=processing_config.sfreq, notch_freq=processing_config.notch_freq
            ),
        ),
        (
            "bandpass_filter",
            BandPassFilter(
                sfreq=processing_config.sfreq,
                low_freq=processing_config.bandpass_low,
                high_freq=processing_config.bandpass_high,
                order=processing_config.filter_order,
            ),
        ),
        (
            "temporal_crop",
            TemporalCrop(
                tmin=processing_config.tmin,
                tmax=processing_config.tmax,
                sfreq=processing_config.sfreq,
            ),
        ),
        ("channel_normalizer", ChannelWiseNormalizer(axis=(0, 2))),
        ("reshaper", EEGReshaper(target_shape="keep")),
        ("unsqueeze", FunctionTransformer(unsqueeze_for_eeg)),
    ]
    if not test:
        steps.insert(
            3,
            (
                "artifact_removal",
                StatisticalArtifactRemoval(
                    z_threshold=processing_config.z_threshold, method="iqr"
                ),
            ),
        )

    return Pipeline(steps)


def preprocess_and_save(
    train_dataset: Dataset,
    val_dataset: Dataset,
    task_type: str,
    processing_config: ProcessingConfig,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),  # type: ignore
        shuffle=False,
    )
    train_data, train_labels = next(iter(train_loader))
    train_data, train_labels = train_data.numpy(), train_labels.numpy()

    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),  # type: ignore
        shuffle=False,
    )
    val_data, val_labels = next(iter(val_loader))
    val_data, val_labels = val_data.numpy(), val_labels.numpy()

    train_pipeline = create_eeg_pipeline(
        task_type=task_type, processing_config=processing_config
    )
    print("\nTraining Pipeline steps:")
    for i, (name, transformer) in enumerate(train_pipeline.steps):
        print(f"{i + 1}. {name}: {transformer.__class__.__name__}")

    test_pipeline = create_eeg_pipeline(
        task_type=task_type, processing_config=processing_config, test=True
    )
    print("\nTesting Pipeline steps:")
    for i, (name, transformer) in enumerate(test_pipeline.steps):
        print(f"{i + 1}. {name}: {transformer.__class__.__name__}")

    print(f"\nOriginal train data shape: {train_data.shape}")
    print(f"Original validation data shape: {val_data.shape}")

    train_data_transformed = train_pipeline.fit_transform(train_data)
    test_pipeline.fit(train_data)
    clean_indices = train_pipeline.named_steps["artifact_removal"].clean_indices_
    train_labels_transformed = train_labels[clean_indices]
    print(f"\nTransformed train data shape: {train_data_transformed.shape}")
    print(f"Transformed train labels shape: {train_labels_transformed.shape}")

    val_data_transformed = train_pipeline.transform(val_data)
    clean_indices = train_pipeline.named_steps["artifact_removal"].clean_indices_
    val_labels_transformed = val_labels[clean_indices]
    print(f"\nTransformed validation data shape: {val_data_transformed.shape}")
    print(f"Transformed validation labels shape: {val_labels_transformed.shape}")

    print(
        "\nTrain Channel Means (should be zero mean):\n",
        train_data_transformed.mean(axis=(0, 1, 3)),
    )
    print(
        "\nTrain Channel Standard Deviations (should be unit std):\n",
        train_data_transformed.std(axis=(0, 1, 3)),
    )
    print(
        "\nValidation Channel Means (should be zero mean):\n",
        val_data_transformed.mean(axis=(0, 1, 3)),
    )
    print(
        "\nValidation Channel Standard Deviations (should be unit std or close to it):\n",
        val_data_transformed.std(axis=(0, 1, 3)),
    )

    output_dir = PROCESSED_DATA_DIR / task_type.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_data.npy", train_data_transformed)
    np.save(output_dir / "train_labels.npy", train_labels_transformed)
    np.save(output_dir / "validation_data.npy", val_data_transformed)
    np.save(output_dir / "validation_labels.npy", val_labels_transformed)
    print(f"\n✅ Processed data successfully saved to '{output_dir}'")

    pipeline_dir = TRAINING_STATS_PATH / task_type.upper()
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(train_pipeline, pipeline_dir / "train_pipeline.pkl")
    joblib.dump(test_pipeline, pipeline_dir / "test_pipeline.pkl")
    print(f"\n✅ Preprocessing pipelines successfully saved to '{pipeline_dir}'")
