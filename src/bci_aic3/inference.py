from pathlib import Path

from sklearn.base import BaseEstimator
import torch
from torch.utils.data import DataLoader, TensorDataset

from bci_aic3.config import load_processing_config
from bci_aic3.data import BCIDataset
from bci_aic3.paths import (
    LABEL_MAPPING_PATH,
    MI_CONFIG_PATH,
    MI_RUNS_DIR,
    RAW_DATA_DIR,
    REVERSE_LABEL_MAPPING_PATH,
    SSVEP_CONFIG_PATH,
    TRAINING_STATS_PATH,
)
from bci_aic3.preprocess import MIBCIPreprocessor
from bci_aic3.util import (
    load_model,
    read_json_to_dict,
)


def load_and_preprocess_for_inference(
    csv_file: str, base_path: str | Path, task_type: str
) -> TensorDataset:
    # determine config_path and preprocessor for task_type
    config_path = None
    preprocessor = BaseEstimator
    if task_type == "MI":
        config_path = MI_CONFIG_PATH
        preprocessor = MIBCIPreprocessor
    elif task_type == "SSVEP":
        config_path = SSVEP_CONFIG_PATH
        # preprocessor = SSVEPPreprocessor
    else:
        raise (
            ValueError(
                f"Invalid task_type: {task_type}.\nValid task_type (MI) or (SSVEP)"
            )
        )

    # load data
    test = BCIDataset(
        csv_file=csv_file,
        base_path=base_path,
        task_type=task_type,
        split="test",
        label_mapping=read_json_to_dict(LABEL_MAPPING_PATH),
    )

    # create data loader
    data_loader = DataLoader(test, batch_size=len(test), shuffle=False)
    data_batch = next(iter(data_loader))
    data = data_batch.numpy()

    processing_config = load_processing_config(config_path)

    # load preprocessor
    preprocessor = preprocessor.load(TRAINING_STATS_PATH, config=processing_config)  # type: ignore

    # apply transformation
    processed_data = preprocessor.transform(X=data)

    test_tensor = torch.from_numpy(processed_data).float()

    return TensorDataset(test_tensor)


def create_inference_data_loader(
    test_dataset: TensorDataset, batch_size: int | None = None
):
    if batch_size is None:
        batch_size = len(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def predict_batch(
    model, data_loader, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Make predictions on a batch of data."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch[0]
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
        predictions = [reverse_label_mapping[str(p)] for p in predictions]

    return predictions


def main():
    # Config paths
    model_path = MI_RUNS_DIR / "model_scripted.pt"
    task_type = "MI"

    # load model
    model = load_model(model_path=model_path)

    # Run inference
    predictions = make_inference(
        model=model,
        csv_file="test.csv",
        base_path=RAW_DATA_DIR,
        task_type=task_type,
        reverse_mapping=True,
    )

    print(predictions)


if __name__ == "__main__":
    main()
