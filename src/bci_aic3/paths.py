# src/paths.py

from pathlib import Path


def get_project_root():
    """
    Finds the project root. (The extra cases are to handle kaggle environments)

    This function is designed for the Kaggle environment where a project
    repository is cloned into the `/kaggle/working` directory. It first checks
    if the current directory is the project root, and if not, it searches
    one level deep in the subdirectories.

    Returns:
        Path: The path to the project root.
                      Falls back to the current working directory if no markers are found.
    """
    current_path = Path.cwd()  # In Kaggle, this is typically /kaggle/working

    # Define the markers that identify the root of your project
    project_markers = ["pyproject.toml", "src", ".git", "uv.lock"]

    # Case 1: The current directory is the project root
    for marker in project_markers:
        if (current_path / marker).exists():
            return current_path

    # Case 2: The project is in an immediate subdirectory (the common case)
    # e.g., current_path is /kaggle/working, project is /kaggle/working/repo_name
    for child in current_path.iterdir():
        if child.is_dir():
            for marker in project_markers:
                if (child / marker).exists():
                    # This subdirectory is the project root
                    return child

    # Fallback: If no project structure is found, return the starting directory.
    print(
        f"Warning: Could not find project root markers {project_markers}. Falling back to {current_path}."
    )
    return current_path


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
