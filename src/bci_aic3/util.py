# src/util.py

import json
import os
from pathlib import Path
from typing import Dict

import torch
from torch.nn.utils import parametrize
from torcheeg.models import (
    FBCCNN,
    MTCNN,
    ATCNet,
    EEGNet,
    FBCNet,
    TSCeption,
    FBMSNet,
)


def get_model_class(model_name: str) -> torch.nn.Module:
    models = {
        "FBCCNN": FBCCNN,
        "MTCNN": MTCNN,
        "ATCNet": ATCNet,
        "EEGNet": EEGNet,
        "FBCNet": FBCNet,
        "TSCeption": TSCeption,
        "FBMSNet": FBMSNet,
    }
    return models[model_name]


def ensure_base_path(base_path: str | Path) -> Path:
    # Convert base_path to Path object if it's a string
    if isinstance(base_path, str):
        base_path = Path(base_path)

    # Raise error if path does not exist
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base path '{base_path}' does not exist. This should point to your raw data directory."
        )

    # Raise error if path is not a directory
    if not base_path.is_dir():
        raise NotADirectoryError(
            f"Expected directory at '{base_path}', but it's not a directory. This should point to your raw data directory."
        )

    return base_path


def rec_cpu_count() -> int:
    """Returns recommended cpu count based on machine and a simple heuristic"""
    cpu_count = os.cpu_count()

    if cpu_count is None:
        return 4

    return min(cpu_count // 2, 8)


def read_json_to_dict(file_path) -> dict:
    """Reads a JSON file and returns a dictionary."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def apply_normalization(x, channel_means, channel_stds):
    # x: Tensor of shape (C, T) or (N, C, T)
    return (x - channel_means[:, None]) / channel_stds[:, None]


# Remove all parametrizations from the model (necessary to save as torchscript)
def remove_parametrizations(model):
    for name, module in model.named_modules():
        if hasattr(module, "parametrizations"):
            # Remove all parametrizations for this module
            for param_name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, param_name)
    return model


def save_model(model: torch.nn.Module, save_path: Path) -> None:
    """
    Saves the model's state dictionary to the specified path.

    This is a general-purpose function for saving the learned weights
    of any PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to save (can be nn.Module,
                           pl.LightningModule, etc.).
        save_path (Path): The file path to save the state_dict to.
                          Conventionally ends with .pt or .pth.
    """
    print(f"Saving model weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model weights saved successfully.")


def load_model(
    model_class: type[torch.nn.Module], model_path: Path, *args, **kwargs
) -> torch.nn.Module:
    """
    Loads a model's state dictionary from a file.

    This function first instantiates the model and then loads the saved
    state dictionary into it.

    Args:
        model_class (type[torch.nn.Module]): The class of the model to be loaded.
        model_path (Path): The path to the saved model's state_dict.
        *args: Variable length argument list for the model's constructor.
        **kwargs: Arbitrary keyword arguments for the model's constructor.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    # Instantiate the model with its required arguments
    model = model_class(*args, **kwargs)

    # Load the state dictionary
    state_dict = torch.load(model_path)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    return model


# Helper functions for managing training statistics
def save_training_stats(stats: Dict[str, torch.Tensor | None], save_path: Path):
    """Save training statistics to disk."""
    torch.save(stats, save_path)


def load_training_stats(load_path: Path) -> Dict[str, torch.Tensor]:
    """Load training statistics from disk."""
    return torch.load(load_path, weights_only=False)
