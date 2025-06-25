from pathlib import Path

import joblib
import mne
import numpy as np
from mne.decoding import CSP
from mne.preprocessing import ICA
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader

from bci_aic3.config import ProcessingConfig
from bci_aic3.data import BCIDataset
from bci_aic3.paths import (
    PROCESSED_DATA_DIR,
    TRAINING_STATS_PATH,
)


class MIBCIPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessing pipeline for Motor Imagery BCI.

    This pipeline applies the following steps:
    1. Converts NumPy array to MNE EpochsArray.
    2. Applies notch and bandpass filters.
    3. Fits and applies ICA for artifact removal.
    4. Crops the epochs to the time window of interest.
    5. Fits and applies Common Spatial Patterns (CSP).
    6. Calculates and applies channel-wise normalization (mean/std).

    The fitted components (ICA, CSP, normalization stats) can be saved and loaded.
    """

    def __init__(self, config: ProcessingConfig):
        """Initializes the preprocessor with a given configuration."""
        self.config = config
        self.ica = None
        self.csp = None
        self.info = None
        self.mean_ = None  # To store mean for normalization
        self.std_ = None  # To store std for normalization

    def _create_mne_epochs(self, X: np.ndarray) -> mne.EpochsArray:
        """Creates an MNE EpochsArray object from a NumPy array."""
        if self.info is None:
            self.info = mne.create_info(
                ch_names=self.config.ch_names, sfreq=self.config.sfreq, ch_types="eeg"
            )
            montage = mne.channels.make_standard_montage("standard_1020")
            self.info.set_montage(montage, on_missing="ignore")
        return mne.EpochsArray(X, self.info, tmin=0, verbose=False)

    def _apply_filters(self, epochs: mne.EpochsArray) -> mne.EpochsArray:
        """Applies notch and bandpass filters to the data array."""
        data = epochs.get_data(copy=True)
        data_notched = mne.filter.notch_filter(
            data,
            Fs=self.config.sfreq,
            freqs=self.config.notch_freq,
            method="iir",
            verbose=False,
        )
        b, a = butter(  # type: ignore
            self.config.filter_order,
            [self.config.bandpass_low, self.config.bandpass_high],
            btype="bandpass",
            fs=self.config.sfreq,
        )
        data_bandpassed = filtfilt(b, a, data_notched, axis=-1)
        return mne.EpochsArray(
            data_bandpassed, epochs.info, tmin=epochs.tmin, verbose=False
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits ICA, CSP, and computes normalization statistics from the training data.

        Args:
            X: EEG data, shape (n_epochs, n_channels, n_times)
            y: Labels, shape (n_epochs,)
        """
        # Steps 1-2: Create MNE object and apply filters
        epochs = self._create_mne_epochs(X)
        epochs_filtered = self._apply_filters(epochs)

        # Step 3: Fit ICA
        self.ica = ICA(
            n_components=self.config.ica_n_components,
            random_state=self.config.ica_random_state,
            max_iter="auto",
        )
        self.ica.fit(epochs_filtered)
        epochs_ica = self.ica.apply(epochs_filtered.copy(), verbose=False)

        # Step 4: Crop
        epochs_cropped = epochs_ica.copy().crop(
            tmin=self.config.tmin, tmax=self.config.tmax, include_tmax=False
        )

        # Step 5: Fit CSP
        self.csp = CSP(
            n_components=self.config.n_csp_components,
            reg=None,
            log=None,
            norm_trace=False,
            transform_into="csp_space",
        )
        csp_data = self.csp.fit_transform(epochs_cropped.get_data(copy=False), y)

        # Step 6: Calculate normalization statistics from the CSP-transformed training data
        self.mean_ = np.mean(csp_data, axis=(0, 2), keepdims=True)
        self.std_ = np.std(csp_data, axis=(0, 2), keepdims=True)

        # Add a small epsilon to std to avoid division by zero
        self.std_[self.std_ == 0] = 1e-6

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the learned transformations (ICA, CSP, Normalization) to the data.
        """
        if (
            self.ica is None
            or self.csp is None
            or self.mean_ is None
            or self.std_ is None
        ):
            raise RuntimeError(
                "The preprocessor has not been fitted. Call fit() or load() first."
            )

        # Steps 1-4: Create MNE object, filter, apply ICA, crop
        epochs = self._create_mne_epochs(X)
        epochs_filtered = self._apply_filters(epochs)
        epochs_ica = self.ica.apply(epochs_filtered, verbose=False)
        epochs_cropped = epochs_ica.crop(
            tmin=self.config.tmin, tmax=self.config.tmax, include_tmax=False
        )

        # Step 5: Apply CSP
        csp_data = self.csp.transform(epochs_cropped.get_data(copy=False))

        # Step 6: Apply Normalization using stored stats
        normalized_data = (csp_data - self.mean_) / self.std_

        return normalized_data.astype(np.float32)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it."""
        self.fit(X, y)  # type: ignore
        return self.transform(X)  # type: ignore

    def save(self, path: Path):
        """Saves the fitted components to a directory."""
        if any(
            attr is None
            for attr in [self.ica, self.csp, self.info, self.mean_, self.std_]
        ):
            raise RuntimeError("The preprocessor has not been fitted yet. Cannot save.")

        path.mkdir(parents=True, exist_ok=True)
        self.ica.save(path / "ica-ica.fif", overwrite=True)  # type: ignore
        joblib.dump(self.csp, path / "csp.gz")
        joblib.dump(self.info, path / "info.gz")
        joblib.dump({"mean": self.mean_, "std": self.std_}, path / "norm_stats.gz")
        print(f"✅ Preprocessor components successfully saved to '{path}'")

    @classmethod
    def load(cls, path: Path, config: ProcessingConfig):
        """Loads a pre-fitted preprocessor from a directory."""
        instance = cls(config)
        instance.ica = mne.preprocessing.read_ica(path / "ica-ica.fif")
        instance.csp = joblib.load(path / "csp.gz")
        instance.info = joblib.load(path / "info.gz")
        norm_stats = joblib.load(path / "norm_stats.gz")
        instance.mean_ = norm_stats["mean"]
        instance.std_ = norm_stats["std"]
        print(f"✅ Preprocessor components successfully loaded from '{path}'")
        return instance


def preprocessing_pipeline(
    train_dataset: BCIDataset,
    validation_dataset: BCIDataset,
    task_type: str,
    processing_config: ProcessingConfig,
):
    """The main pipeline to preprocess data and save the processor."""
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    train_data, train_labels = next(iter(train_loader))
    val_loader = DataLoader(validation_dataset, batch_size=len(validation_dataset))
    val_data, val_labels = next(iter(val_loader))

    train_data, train_labels = train_data.numpy(), train_labels.numpy()
    val_data, val_labels = val_data.numpy(), val_labels.numpy()

    preprocessor = MIBCIPreprocessor(processing_config)
    print("Fitting preprocessor and transforming training data...")
    train_data_processed = preprocessor.fit_transform(train_data, train_labels)
    print("Done.")

    print("Saving preprocessor state...")

    stats_saving_path = TRAINING_STATS_PATH / task_type
    stats_saving_path.mkdir(parents=True, exist_ok=True)
    preprocessor.save(stats_saving_path)

    print("Transforming validation data with the fitted preprocessor...")
    val_data_processed = preprocessor.transform(val_data)
    print("Done.")

    output_dir = PROCESSED_DATA_DIR / task_type.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_data.npy", train_data_processed)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "validation_data.npy", val_data_processed)
    np.save(output_dir / "validation_labels.npy", val_labels)
    print(f"✅ Processed data saved to '{output_dir}'")


def main():
    print("help me")


# --- Usage Example ---
if __name__ == "__main__":
    main()
