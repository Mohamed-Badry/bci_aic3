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
LABEL_MAPPING_PATH = CONFIG_DIR / "label_mapping_str_to_int.json"
REVERSE_LABEL_MAPPING_PATH = CONFIG_DIR / "label_mapping_str_to_int.json"


# data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
