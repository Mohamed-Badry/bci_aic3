# src/paths.py

from pathlib import Path


def get_project_root():
    """
    Finds the project root by searching upwards from this script's location.

    The project root is unequivocally identified by the presence of marker files
    that only exist at the top level of the project, such as '.git' or
    'pyproject.toml'. This ensures the function is 100% reliable.

    Returns:
        pathlib.Path: The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If no project root is found. This function will not
                           return a potentially incorrect path.
    """
    # Use markers that are guaranteed to be at the project's top level.
    # 'src' is removed because it would cause the function to incorrectly
    # identify the 'src' directory itself as the root.
    markers = [".git", "pyproject.toml"]

    # Start the search from the location of this file.
    # This is more reliable than using the current working directory.
    current_path = Path(__file__).resolve()

    while current_path.parent != current_path:  # Stop at the filesystem root
        for marker in markers:
            if (current_path / marker).exists():
                return current_path  # Project root found
        current_path = current_path.parent  # Move up one level

    raise FileNotFoundError(
        "Could not find project root. Traversed up from "
        f"'{Path(__file__).resolve()}' but no marker "
        f"({', '.join(markers)}) was found."
    )


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
MI_TRAINING_STATS_PATH = TRAINING_STATS_PATH / "MI"
SSVEP_TRAINING_STATS_PATH = TRAINING_STATS_PATH / "SSVEP"

# data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# model runs
RUNS_DIR = PROJECT_ROOT / "run"
MI_RUNS_DIR = RUNS_DIR / "MI"
SSVEP_RUNS_DIR = RUNS_DIR / "SSVEP"

# best model checkpoint
BEST_MODELS_CHECKPOINT = PROJECT_ROOT / "best"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
