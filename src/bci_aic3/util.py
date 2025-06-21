# src/util.py

import os
import json
import torch
from pathlib import Path
from typing import Dict


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


def load_model(model_path, model_class, model_kwargs, device, optim, learning_rate):
    model = model_class(**model_kwargs).to(device)
    optimizer = optim(model.parameters(), lr=learning_rate)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer


def save_model(
    save_path: str | Path, epoch, model, optimizer, loss: float, f1_score: float
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "f1_score": f1_score,
        },
        save_path,
    )


def apply_normalization(x, channel_means, channel_stds):
    # x: Tensor of shape (C, T) or (N, C, T)
    return (x - channel_means[:, None]) / channel_stds[:, None]


# Helper functions for managing training statistics
def save_training_stats(stats: Dict[str, torch.Tensor | None], save_path: Path):
    """Save training statistics to disk."""
    torch.save(stats, save_path)


def load_training_stats(load_path: Path) -> Dict[str, torch.Tensor]:
    """Load training statistics from disk."""
    return torch.load(load_path)
