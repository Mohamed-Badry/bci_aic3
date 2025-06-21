# src/paths.py

from pathlib import Path


def get_project_root() -> Path:
    """Find project root by looking for a marker file/folder."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find project root")


PROJECT_ROOT = get_project_root()


# configs
CONFIG_DIR = PROJECT_ROOT / "configs"
MI_CONFIG_PATH = CONFIG_DIR / "mi_config.yaml"
SSVEP_CONFIG_PATH = CONFIG_DIR / "ssvep_config.yaml"

LABEL_MAPPING_PATH = CONFIG_DIR / "label_mapping_str_to_int.json"
REVERSE_LABEL_MAPPING_PATH = CONFIG_DIR / "label_mapping_str_to_int.json"

# training statistics
TRAINING_STATS_PATH = PROJECT_ROOT / "training_stats"

# data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# models
MODELS_DIR = PROJECT_ROOT / "models"

# checkpoints
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MI_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "MI"
SSVEP_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "SSVEP"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
