from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from bci_aic3.paths import (
    CONFIG_DIR,
    RAW_DATA_DIR,
    REVERSE_LABEL_MAPPING_PATH,
    RUNS_DIR,
)
from bci_aic3.preprocess import load_and_preprocess_for_inference
from bci_aic3.util import load_model, read_json_to_dict


def create_inference_data_loader(
    test_dataset: TensorDataset, batch_size: int | None = None
):
    if batch_size is None:
        batch_size = len(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def predict_batch(model, data_loader, device):
    """Make predictions on a batch of data."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds.tolist())

    return predictions


def make_inference(
    model,
    csv_file: str,
    base_path: str | Path,
    task_type: str,
    reverse_mapping: bool = True,
    batch_size: int | None = None,
):
    """Simple batch inference maintaining order."""

    # Convert base_path to Path object if it's a string
    if isinstance(base_path, str):
        base_path = Path(base_path)

    # load and preprocess test data
    test_dataset = load_and_preprocess_for_inference(
        csv_file=csv_file, base_path=base_path, task_type=task_type
    )

    if batch_size is None:
        batch_size = len(test_dataset)

    test_loader = create_inference_data_loader(
        test_dataset=test_dataset, batch_size=batch_size
    )

    predictions = predict_batch(
        model=model,
        data_loader=test_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Turn labels from ints back to strings
    if reverse_mapping:
        reverse_label_mapping = read_json_to_dict(REVERSE_LABEL_MAPPING_PATH)
        predictions = [reverse_label_mapping[str(p.item())] for p in predictions]

    return predictions


def main():
    # Config paths
    ssvep_config_path = CONFIG_DIR / "ssvep_model_config.json"
    mi_config_path = CONFIG_DIR / "mi_model_config.json"
    task_type = "MI"

    # TODO: implement model loading
    model_dir_path = None
    model = load_model()

    # Run inference
    predictions = make_inference(
        model=model,
        csv_file="test.csv",
        base_path=RAW_DATA_DIR,
        task_type=task_type,
        reverse_mapping=True,
    )

    # Create submission
    submission_df = pd.DataFrame(
        {"Id": range(len(predictions)), "Predicted": predictions}
    )

    submission_df.to_csv("submission.csv", index=False)
    print(f"Submission saved with {len(predictions)} predictions!")


if __name__ == "__main__":
    main()
