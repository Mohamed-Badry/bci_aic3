import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from bci_aic3.data import BCIDataset
from bci_aic3.models.eegnet import EEGNet
from bci_aic3.paths import (
    CONFIG_DIR,
    LABEL_MAPPING_PATH,
    RUNS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REVERSE_LABEL_MAPPING_PATH,
)
from bci_aic3.util import load_model, read_json_to_dict


def load_models(ssvep_config, mi_config):
    """Load both models."""
    ssvep_model = load_model()

    mi_model = load_model()

    return ssvep_model, mi_model


def predict_batch(model, data_loader, device):
    """Make predictions on a batch of data."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds.tolist())

    return predictions


def make_inference(test_csv_path, ssvep_config_path, mi_config_path, batch_size=32):
    """Simple batch inference maintaining order."""

    # Load configs and models
    ssvep_config = read_json_to_dict(ssvep_config_path)
    mi_config = read_json_to_dict(mi_config_path)
    label_mapping = read_json_to_dict(LABEL_MAPPING_PATH)

    print("Loading models...")
    ssvep_model, mi_model = load_models(ssvep_config, mi_config)
    device = ssvep_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Read test data
    test_df = pd.read_csv(test_csv_path)
    print(f"Processing {len(test_df)} samples...")

    # Initialize results list to maintain order
    all_predictions = [None] * len(test_df)

    # Process SSVEP samples
    ssvep_indices = test_df[test_df["task_type"] == "SSVEP"].index.tolist()
    if ssvep_indices:
        print(f"Processing {len(ssvep_indices)} SSVEP samples...")

        ssvep_dataset = BCIDataset(
            csv_file=test_csv_path,
            base_path=RAW_DATA_DIR,
            task_type="SSVEP",
            split="test",
            label_mapping=label_mapping,
        )

        ssvep_loader = DataLoader(ssvep_dataset, batch_size=batch_size, shuffle=False)
        ssvep_preds = predict_batch(ssvep_model, ssvep_loader, device)

        # Put predictions back in original order
        for i, pred in zip(ssvep_indices, ssvep_preds):
            all_predictions[i] = pred

    # Process MI samples
    mi_indices = test_df[test_df["task_type"] == "MI"].index.tolist()
    if mi_indices:
        print(f"Processing {len(mi_indices)} MI samples...")

        mi_dataset = BCIDataset(
            csv_file=test_csv_path,
            base_path=RAW_DATA_DIR,
            task_type="MI",
            split="test",
            label_mapping=label_mapping,
        )

        mi_loader = DataLoader(mi_dataset, batch_size=batch_size, shuffle=False)
        mi_preds = predict_batch(mi_model, mi_loader, device)

        # Put predictions back in original order
        for i, pred in zip(mi_indices, mi_preds):
            all_predictions[i] = pred

    return all_predictions


def main():
    # Config paths
    ssvep_config_path = CONFIG_DIR / "ssvep_model_config.json"
    mi_config_path = CONFIG_DIR / "mi_model_config.json"

    # Run inference
    predictions = make_inference(
        test_csv_path="test.csv",
        ssvep_config_path=str(ssvep_config_path),
        mi_config_path=str(mi_config_path),
        batch_size=32,
    )

    # Create submission
    submission_df = pd.DataFrame(
        {"Id": range(len(predictions)), "Predicted": predictions}
    )

    submission_df.to_csv("submission.csv", index=False)
    print(f"Submission saved with {len(predictions)} predictions!")


if __name__ == "__main__":
    main()
