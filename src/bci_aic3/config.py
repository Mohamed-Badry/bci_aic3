import yaml
from dataclasses import dataclass, field
from .paths import CONFIG_DIR
from typing import List


@dataclass
class ModelConfig:
    name: str
    task_type: str
    num_classes: int
    sequence_length: int
    num_channels: int
    n_csp_components: int  # Number of CSP components to keep (usually 4-8)
    new_sequence_length: (
        int  # this is the sequence length calculated after preprocessing
    )


def load_model_config(path):
    """Load model config from YAML file."""
    with open(path) as f:
        config_dict = yaml.safe_load(f)
        processing_config = config_dict["preprocessing"]

        config_dict["model"]["new_sequence_length"] = int(
            (processing_config["tmax"] - processing_config["tmin"])
            * processing_config["sfreq"]
        )

    return ModelConfig(**config_dict["model"])


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    patience: int


def load_training_config(path):
    """Load training config from YAML file."""
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict["training"])


@dataclass
class ProcessingConfig:
    """Configuration for the MI-BCI preprocessing pipeline."""

    # Data parameters
    sfreq: int

    # Filtering parameters
    notch_freq: float
    bandpass_low: float
    bandpass_high: float
    filter_order: int

    # Epoching and Cropping
    tmin: float  # Start time of the MI task relative to cue
    tmax: float  # End time of the MI task (e.g., 3-second window)

    # ICA parameters
    ica_n_components: int  # Use all channels for ICA
    ica_random_state: int

    # CSP parameters
    n_csp_components: int  # Number of CSP components to keep (usually 4-8)

    ch_names: List[str] = field(
        default_factory=lambda: ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]
    )


def load_processing_config(path):
    """Load preprocessing settings from YAML file."""
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return ProcessingConfig(**config_dict["preprocessing"])


def main():
    model_config = load_model_config(CONFIG_DIR / "mi_config.yaml")
    print(model_config)

    training_config = load_training_config(CONFIG_DIR / "mi_config.yaml")
    print(training_config)

    processing_settings = load_processing_config(CONFIG_DIR / "mi_config.yaml")
    print(processing_settings)


if __name__ == "__main__":
    main()
