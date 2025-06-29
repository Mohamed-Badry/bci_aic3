from typing import List, Tuple, Union

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


class MNEICA(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for non-destructive EEG artifact removal using MNE's ICA.

    This transformer fits ICA on the provided data, allows for the exclusion of artifactual
    components, and then reconstructs the signal without these components.

    Parameters
    ----------
    sfreq : float
        The sampling frequency of the EEG data.
    n_components : int | float | None, default=None
        The number of principal components to use for ICA.
        If int, it must be <= n_channels.
        If float (0 < n_components < 1), it selects the number of components that
        explain at least `n_components` of the variance.
        If None, all components are used.
    random_state : int | None, default=None
        The seed for the random number generator for reproducibility.
    exclude : list of int, default=None
        A list of IC indices to exclude. If None, the transformer will need to be
        fit and the `exclude` attribute set manually before transforming data.
    """

    def __init__(
        self,
        sfreq: float,
        n_components: Union[int, float, None] = None,
        random_state: int = 42,
        exclude: list = None,  # type: ignore
    ):
        self.sfreq = sfreq
        self.n_components = n_components
        self.random_state = random_state
        self.exclude = exclude
        self.ica_ = None

    def fit(self, X, y=None):
        """
        Fits the ICA model to the EEG data.

        Args:
            X: EEG data of shape (n_epochs, n_channels, n_timepoints)
            y: Not used, for compatibility with sklearn API.

        Returns:
            self: The fitted transformer instance.
        """
        # MNE works with data in (n_channels, n_timepoints) format.
        # We can concatenate the epochs to fit the ICA model.
        X_concat = np.concatenate(X, axis=1)

        # Create an MNE Raw object to work with MNE's ICA
        ch_names = [f"EEG {i + 1}" for i in range(X.shape[1])]
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(X_concat, info, verbose=False)

        # High-pass filter the data for better ICA performance
        raw.filter(l_freq=1.0, h_freq=None, verbose=False)

        self.ica_ = mne.preprocessing.ICA(
            n_components=self.n_components,
            random_state=self.random_state,
            verbose=False,
        )
        self.ica_.fit(raw, verbose=False)

        return self

    def transform(self, X):
        """
        Applies the fitted ICA to remove artifacts from the EEG data.

        Args:
            X: EEG data of shape (n_epochs, n_channels, n_timepoints)

        Returns:
            Filtered EEG data of the same shape.
        """
        if self.ica_ is None:
            raise RuntimeError(
                "The ICA model has not been fitted yet. Call fit() first."
            )

        if self.exclude is None:
            print(
                "Warning: The 'exclude' attribute is not set. No components will be removed."
            )
            return X

        X_transformed = np.zeros_like(X)
        ch_names = [f"EEG {i + 1}" for i in range(X.shape[1])]
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="eeg")

        for i, epoch in enumerate(X):
            raw_epoch = mne.io.RawArray(epoch, info, verbose=False)
            self.ica_.apply(raw_epoch, exclude=self.exclude, verbose=False)
            X_transformed[i] = raw_epoch.get_data()

        return X_transformed

    def plot_components(self, **kwargs):
        """
        Plot the ICA components.

        This is a helper function to visualize the components to decide which to exclude.
        """
        if self.ica_ is None:
            raise RuntimeError(
                "The ICA model has not been fitted yet. Call fit() first."
            )
        self.ica_.plot_components(**kwargs)

    def plot_sources(self, X, **kwargs):
        """
        Plot the time course of the ICA sources.

        Args:
            X: EEG data of shape (n_epochs, n_channels, n_timepoints)
        """
        if self.ica_ is None:
            raise RuntimeError(
                "The ICA model has not been fitted yet. Call fit() first."
            )

        X_concat = np.concatenate(X, axis=1)
        info = mne.create_info(
            ch_names=[f"EEG {i + 1}" for i in range(X.shape[1])],
            sfreq=self.sfreq,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(X_concat, info, verbose=False)
        self.ica_.plot_sources(raw, **kwargs)


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
    task_type: str,
    processing_config: ProcessingConfig,
    ica_exclude: List[int],
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
        (
            "artifact_removal",
            MNEICA(
                sfreq=processing_config.sfreq,
                n_components=processing_config.ica_components,
                random_state=42,
                exclude=processing_config.ica_exclude,
            ),
        ),
        ("channel_wise", ChannelWiseNormalizer()),
        ("reshaper", EEGReshaper(target_shape="keep")),
        ("unsqueeze", FunctionTransformer(unsqueeze_for_eeg)),
    ]

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
        task_type=task_type,
        processing_config=processing_config,
        ica_exclude=[0, 2],
    )
    print("\nTraining Pipeline steps:")
    for i, (name, transformer) in enumerate(train_pipeline.steps):
        print(f"{i + 1}. {name}: {transformer.__class__.__name__}")

    print(f"\nOriginal train data shape: {train_data.shape}")
    print(f"Original validation data shape: {val_data.shape}")

    train_data_transformed = train_pipeline.fit_transform(train_data)
    train_labels_transformed = train_labels
    print(f"\nTransformed train data shape: {train_data_transformed.shape}")
    print(f"Transformed train labels shape: {train_labels_transformed.shape}")

    val_data_transformed = train_pipeline.transform(val_data)
    val_labels_transformed = val_labels
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
    joblib.dump(train_pipeline, pipeline_dir / "transform_pipeline.pkl")
    print(f"\n✅ Preprocessing pipelines successfully saved to '{pipeline_dir}'")
