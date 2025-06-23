# src/util.py

import json
import os
from pathlib import Path
from typing import Dict

import torch
from torch.nn.utils import parametrize


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


# Save model as torchscript
def save_model(model, save_path: Path) -> None:
    remove_parametrizations(model)
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, save_path)


# Load torchscript model
def load_model(model_path: Path) -> type[torch.nn.Module]:
    loaded_model = torch.jit.load(model_path)
    return loaded_model


# Helper functions for managing training statistics
def save_training_stats(stats: Dict[str, torch.Tensor | None], save_path: Path):
    """Save training statistics to disk."""
    torch.save(stats, save_path)


def load_training_stats(load_path: Path) -> Dict[str, torch.Tensor]:
    """Load training statistics from disk."""
    return torch.load(load_path, weights_only=False)
