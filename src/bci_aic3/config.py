import yaml
from dataclasses import dataclass, field
from .paths import CONFIG_DIR
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    name: str
    task_type: str
    num_classes: int
    sequence_length: int
    num_channels: int
    new_sequence_length: (
        int  # this is the sequence length calculated after preprocessing
    )
    params: Dict[str, Any] = field(default_factory=dict)  # All other params


def load_model_config(path):
    """Load model config from YAML file."""
    with open(path) as f:
        config_dict = yaml.safe_load(f)

    processing_config = config_dict["preprocessing"]
    model_dict = config_dict["model"].copy()

    model_dict["new_sequence_length"] = int(
        (processing_config["tmax"] - processing_config["tmin"])
        * processing_config["sfreq"]
    )

    # Get model-specific params
    model_name = model_dict["name"].lower()
    model_dict["params"] = config_dict["model"].get(model_name, {})

    # Remove all model-specific sections (keep only base fields)
    base_fields = {
        "name",
        "task_type",
        "num_classes",
        "sequence_length",
        "num_channels",
        "new_sequence_length",
        "params",
    }
    model_dict = {k: v for k, v in model_dict.items() if k in base_fields}

    return ModelConfig(**model_dict)


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    scheduler_patience: int
    factor: float


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
    tmin: float
    tmax: float

    z_threshold: float

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

    model_kwargs = {
        "num_electrodes": model_config.num_channels,
        "chunk_size": model_config.new_sequence_length,
        "num_classes": model_config.num_classes,
        **model_config.params,
    }

    print(model_kwargs)


if __name__ == "__main__":
    main()
