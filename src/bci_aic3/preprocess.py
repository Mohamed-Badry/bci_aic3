import numpy as np
import mne
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from mne.preprocessing import ICA
from mne.decoding import CSP

from torch.utils.data import DataLoader

from bci_aic3.util import load_training_stats
from bci_aic3.config import ProcessingConfig
from bci_aic3.data import BCIDataset
from bci_aic3.paths import PROCESSED_DATA_DIR, TRAINING_STATS_PATH


class MIBCIPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessing pipeline for Motor Imagery BCI.

    This pipeline applies the following steps:
    1. Converts NumPy array to MNE EpochsArray.
    2. Applies a notch filter to remove powerline noise.
    3. Applies a custom Butterworth bandpass filter.
    4. Fits and applies ICA to remove biological artifacts.
    5. Crops the epochs to the time window of interest.
    6. Fits and applies Common Spatial Patterns (CSP) for feature extraction.

    The output is ready for a classification model like EEGNet.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ica = None
        self.csp = None
        self.info = None

    def _create_mne_epochs(self, X: np.ndarray) -> mne.EpochsArray:
        """Creates an MNE EpochsArray object from a NumPy array."""
        if self.info is None:
            self.info = mne.create_info(
                ch_names=self.config.ch_names, sfreq=self.config.sfreq, ch_types="eeg"
            )
            montage = mne.channels.make_standard_montage("standard_1020")
            self.info.set_montage(montage, on_missing="ignore")

        tmin_original = 0
        return mne.EpochsArray(X, self.info, tmin=tmin_original, verbose=False)

    def _apply_filters(self, epochs: mne.EpochsArray) -> mne.EpochsArray:
        """Applies notch and bandpass filters to the data array."""
        # Get data as a NumPy array to apply filters
        data = epochs.get_data(copy=True)

        # 1. Notch Filter using the top-level MNE function
        data_notched = mne.filter.notch_filter(
            data,
            Fs=self.config.sfreq,
            freqs=self.config.notch_freq,
            method="iir",  # IIR is generally faster for notch
            verbose=False,
        )

        # 2. Bandpass Filter (using a zero-phase Butterworth filter)
        b, a = butter(  # type: ignore
            self.config.filter_order,
            [self.config.bandpass_low, self.config.bandpass_high],
            btype="bandpass",
            fs=self.config.sfreq,
        )

        # Apply the filter forward and backward to each channel and epoch
        data_bandpassed = filtfilt(b, a, data_notched, axis=-1)

        # Create a new EpochsArray with the filtered data
        epochs_filtered = mne.EpochsArray(
            data_bandpassed, epochs.info, tmin=epochs.tmin, verbose=False
        )
        return epochs_filtered

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the ICA and CSP transformers on the training data.

        Args:
            X: EEG data, shape (n_epochs, n_channels, n_times)
            y: Labels, shape (n_epochs,)
        """
        # --- Step 1: Create MNE object and apply basic filters ---
        epochs = self._create_mne_epochs(X)
        epochs_filtered = self._apply_filters(epochs)

        # --- Step 2: Fit ICA for artifact removal ---
        self.ica = ICA(
            n_components=self.config.ica_n_components,
            random_state=self.config.ica_random_state,
            max_iter="auto",
        )
        self.ica.fit(epochs_filtered)

        # --- Step 3: Apply ICA and Crop Epochs ---
        epochs_ica = self.ica.apply(epochs_filtered.copy(), verbose=False)
        epochs_cropped = epochs_ica.copy().crop(
            tmin=self.config.tmin,
            tmax=self.config.tmax,
            include_tmax=False,
        )

        # --- Step 4: Fit CSP ---
        self.csp = CSP(
            n_components=self.config.n_csp_components,
            reg=None,
            log=None,  # Set to False to get the time series, not log-variance
            norm_trace=False,
            transform_into="csp_space",
        )
        self.csp.fit(epochs_cropped.get_data(), y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the learned transformations (ICA, CSP) to the data.

        Args:
            X: EEG data, shape (n_epochs, n_channels, n_times)

        Returns:
            Spatially filtered time-series data, shape (n_epochs, n_csp_components, n_cropped_times)
        """
        if self.ica is None or self.csp is None:
            raise RuntimeError(
                "The preprocessor has not been fitted yet. Call fit() first."
            )

        # --- Step 1 & 2: Create MNE object and apply filters ---
        epochs = self._create_mne_epochs(X)
        epochs_filtered = self._apply_filters(epochs)

        # --- Step 3: Apply fitted ICA ---
        epochs_ica = self.ica.apply(epochs_filtered, verbose=False)

        # --- Step 4: Crop to time window of interest ---
        epochs_cropped = epochs_ica.crop(
            tmin=self.config.tmin, tmax=self.config.tmax, include_tmax=False
        )

        # --- Step 5: Apply fitted CSP to get spatially filtered time series ---
        eegnet_input = self.csp.transform(epochs_cropped.get_data())

        return eegnet_input.astype(np.float32)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)  # type: ignore


def preprocessing_pipeline(
    train_dataset: BCIDataset,
    validation_dataset: BCIDataset,
    task_type: str,
    processing_config: ProcessingConfig,
):
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    train_data, train_labels = next(iter(train_loader))

    val_loader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset), shuffle=False
    )
    val_data, val_labels = next(iter(val_loader))

    train_data = train_data.numpy()
    train_labels = train_labels.numpy()

    val_data = val_data.numpy()
    val_labels = val_labels.numpy()

    preprocessor = MIBCIPreprocessor(processing_config)

    train_data_processed = preprocessor.fit_transform(train_data, train_labels)

    val_data_processed = preprocessor.transform(val_data)

    # Define the directory where the files will be saved
    output_dir = PROCESSED_DATA_DIR / task_type.upper()

    # Create the directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed training data and labels
    processed_train_data_path = output_dir / "train_data.npy"
    processed_labels_path = output_dir / "validation_labels.npy"

    np.save(processed_train_data_path, train_data_processed)
    print(f"Processed data successfully saved at: {processed_train_data_path}")
    np.save(processed_labels_path, train_labels)
    print(f"Processed labels successfully saved at: {processed_labels_path}")

    # Save processed validation data and labels
    processed_val_data_path = output_dir / "validation_data.npy"
    processed_labels_path = output_dir / "validation_labels.npy"

    np.save(processed_val_data_path, val_data_processed)
    print(f"Processed data successfully saved at: {processed_val_data_path}")

    np.save(processed_labels_path, val_labels)
    print(f"Processed labels successfully saved at: {processed_labels_path}")


# --- Usage Example ---
if __name__ == "__main__":
    print("Help me!")
