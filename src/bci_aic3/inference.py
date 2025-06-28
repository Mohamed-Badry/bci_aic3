import joblib

from pathlib import Path


from sklearn.base import BaseEstimator
import torch
from torch.utils.data import DataLoader, TensorDataset
from torcheeg.models import EEGNet

from bci_aic3.preprocess import (  # necessary to load preprocessing pipeline with joblib
    MNENotchFilter,
    BandPassFilter,
    TemporalCrop,
    ChannelWiseNormalizer,
    EEGReshaper,
    StatisticalArtifactRemoval,
    unsqueeze_for_eeg,
)
from bci_aic3.config import load_processing_config
from bci_aic3.data import BCIDataset
from bci_aic3.paths import (
    LABEL_MAPPING_PATH,
    MI_CONFIG_PATH,
    MI_RUNS_DIR,
    MI_TRAINING_STATS_PATH,
    RAW_DATA_DIR,
    REVERSE_LABEL_MAPPING_PATH,
    SSVEP_CONFIG_PATH,
    SSVEP_RUNS_DIR,
    SSVEP_TRAINING_STATS_PATH,
    TRAINING_STATS_PATH,
)
from bci_aic3.train import BCILightningModule
from bci_aic3.util import (
    load_model,
    read_json_to_dict,
)


def load_and_preprocess_for_inference(
    csv_file: str, base_path: str | Path, task_type: str
) -> TensorDataset:
    # determine the test pipeline for the task type
    test_pipeline = None
    if task_type == "MI":
        test_pipeline = joblib.load(MI_TRAINING_STATS_PATH / "test_pipeline.pkl")
    elif task_type == "SSVEP":
        test_pipeline = joblib.load(SSVEP_TRAINING_STATS_PATH / "test_pipeline.pkl")
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

    processed_data = test_pipeline.transform(data_batch.numpy())

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
    model,
    data_loader,
    device,
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
    device: torch.device | str = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
):
    """Simple batch inference maintaining order."""

    # Convert base_path to Path object if it's a string
    if isinstance(base_path, str):
        base_path = Path(base_path)

    # load and preprocess test data
    test_dataset = load_and_preprocess_for_inference(
        csv_file=csv_file, base_path=base_path, task_type=task_type
    )

    test_loader = create_inference_data_loader(
        test_dataset=test_dataset, batch_size=batch_size
    )

    predictions = predict_batch(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    # Turn labels from ints back to strings
    if reverse_mapping:
        reverse_label_mapping = read_json_to_dict(REVERSE_LABEL_MAPPING_PATH)
        predictions = [reverse_label_mapping[str(p)] for p in predictions]

    return predictions


def main():
    # Config paths
    task_type = "SSVEP"
    model_path = (
        SSVEP_RUNS_DIR
        / "../../run/SSVEP/EEGNet-f1-0.5191-20250628_075115/checkpoints/eegnet-ssvep-best-f1-val_f1=0.6345-epoch=32.ckpt"
    )
    eegnet = EEGNet(
        num_electrodes=8,
        chunk_size=1500,
        num_classes=4,
        F1=16,  # number of temporal filters
        F2=32,  # number of spatial filters (F1 * D)
        D=2,  # depth multiplier for depthwise conv
        kernel_1=125,  # ~1 second at 250Hz for temporal conv
        kernel_2=8,  # spatial kernel size (all channels)
        dropout=0.25,
    )
    # load model
    model = BCILightningModule.load_from_checkpoint(model_path, model=eegnet)

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
