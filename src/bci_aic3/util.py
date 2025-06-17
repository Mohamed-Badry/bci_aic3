# src/util.py

import os
import json


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
