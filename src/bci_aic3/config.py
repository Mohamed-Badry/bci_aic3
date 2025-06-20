import yaml
from dataclasses import dataclass
from .paths import CONFIG_DIR


@dataclass
class Config:
    epochs: int
    learning_rate: float
    batch_size: int


def load_training_config(path):
    with open(path) as f:
        config_dict = yaml.safe_load(f)
        # print(config_dict["training"])
    return Config(**config_dict["training"])


def main():
    config = load_training_config(CONFIG_DIR / "basic_config.yaml")
    print(config)


if __name__ == "__main__":
    main()
