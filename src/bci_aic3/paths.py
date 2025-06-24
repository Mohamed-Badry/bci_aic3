# src/paths.py

from pathlib import Path


def get_project_root(target_folder: str = "src") -> Path:
    """
    Simplified version that just goes up until it finds a target folder.

    Args:
        target_folder: Name of the folder to look for (default: "src")
    """
    # Try __file__ first if available
    try:
        start_path = Path(__file__).resolve().parent
    except NameError:
        # Fallback to current working directory
        start_path = Path.cwd()

    # Go up the directory tree until we find the target folder
    current = start_path
    for parent in [current] + list(current.parents):
        if (parent / target_folder).exists():
            return parent

    # If not found, return the starting directory
    return start_path


PROJECT_ROOT = get_project_root()


# configs
CONFIG_DIR = PROJECT_ROOT / "configs"
MI_CONFIG_PATH = CONFIG_DIR / "mi_config.yaml"
SSVEP_CONFIG_PATH = CONFIG_DIR / "ssvep_config.yaml"

LABEL_MAPPING_PATH = CONFIG_DIR / "label_mappings" / "label_mapping.json"
REVERSE_LABEL_MAPPING_PATH = (
    CONFIG_DIR / "label_mappings" / "reverse_label_mapping.json"
)

# training statistics
TRAINING_STATS_PATH = PROJECT_ROOT / "training_stats"

# data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# model runs
RUNS_DIR = PROJECT_ROOT / "run"
MI_RUNS_DIR = RUNS_DIR / "MI"
SSVEP_RUNS_DIR = RUNS_DIR / "SSVEP"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
