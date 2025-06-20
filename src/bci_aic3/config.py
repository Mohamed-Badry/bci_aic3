import yaml
from dataclasses import dataclass
from .paths import CONFIG_DIR


@dataclass
class ModelConfig:
    name: str
    task_type: str
    num_classes: int
    sequence_length: int
    num_channels: int


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    batch_size: int


def load_model_config(path):
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return ModelConfig(**config_dict["model"])


def load_training_config(path):
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict["training"])


def main():
    model_config = load_model_config(CONFIG_DIR / "mi_config.yaml")
    print(model_config)

    training_config = load_training_config(CONFIG_DIR / "mi_config.yaml")
    print(training_config)


if __name__ == "__main__":
    main()
