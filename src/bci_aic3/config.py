import yaml
from dataclasses import dataclass
from .paths import CONFIG_DIR
from typing import Tuple


@dataclass
class ModelConfig:
    name: str
    task_type: str
    num_classes: int
    sequence_length: int
    num_channels: int


def load_model_config(path):
    """Load model config from YAML file."""
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return ModelConfig(**config_dict["model"])


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    batch_size: int


def load_training_config(path):
    """Load training config from YAML file."""
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict["training"])


@dataclass
class ProcessingConfig:
    notch_freq: float
    lfreq: float
    hfreq: float
    baseline: Tuple[float, float]
    tmin: float
    tmax: float
    sfreq: float = 250.0
    scaling_factor: float = 1e-6  # Convert microvolts to volts for MNE


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
