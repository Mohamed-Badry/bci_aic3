# src/train.py

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torchmetrics
from pytorch_lightning import Callback, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader
from torcheeg.models import (
    ATCNet,
    EEGNet,
    FBCNet,
    TSCeption,
)

from bci_aic3.config import (
    ModelConfig,
    TrainingConfig,
    load_model_config,
    load_training_config,
)
from bci_aic3.data import load_processed_data, load_raw_data
from bci_aic3.paths import (
    LABEL_MAPPING_PATH,
    MI_CONFIG_PATH,
    MI_RUNS_DIR,
    PROCESSED_DATA_DIR,
    SSVEP_CONFIG_PATH,
    SSVEP_RUNS_DIR,
)
from bci_aic3.util import (
    read_json_to_dict,
    rec_cpu_count,
    save_model,
)


def get_model_class(model_name: str) -> torch.nn.Module:
    models = {
        "ATCNet": ATCNet,
        "EEGNet": EEGNet,
        "FBCNet": FBCNet,
        "TSCeption": TSCeption,
    }
    return models[model_name]


class BCILightningModule(LightningModule):
    def __init__(
        self,
        model,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = model_config.num_classes
        self.num_channels = model_config.num_channels
        self.sequence_length = model_config.new_sequence_length

        self.training_config = training_config

        # Model
        self.model = model

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics - Lightning handles device placement automatically
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average="macro",
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.num_classes,
            average="macro",
        )

    def forward(self, x):
        # Lightning calls this for inference
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch

        outputs = self(data)  # Use self() instead of self.model() for consistency
        loss = self.criterion(outputs, labels)

        # Metrics
        preds = torch.argmax(outputs, dim=1)
        self.train_accuracy(preds, labels)

        self.train_f1(preds, labels)

        # Lightning automatically logs these
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs, labels)

        # Metrics
        preds = torch.argmax(outputs, dim=1)
        self.val_accuracy(preds, labels)

        self.val_f1(preds, labels)

        # Lightning automatically logs these
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def configure_optimizers(self):  # type: ignore
        optimizer = optim.Adam(self.parameters(), lr=self.training_config.learning_rate)  # type: ignore
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.training_config.factor,
            patience=self.training_config.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def predict_step(self, batch, batch_idx):
        # For inference/prediction
        data, _ = batch
        outputs = self(data)
        return torch.softmax(outputs, dim=1)


def create_raw_data_loaders(
    base_path: str | Path, task_type: str, batch_size: int, num_workers: int
):
    label_mapping = read_json_to_dict(LABEL_MAPPING_PATH)

    # Loading the data
    train, val, test = load_raw_data(
        base_path=base_path, task_type=task_type, label_mapping=label_mapping
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader


def create_processed_data_loaders(
    processed_data_dir: str | Path,
    task_type: str,
    batch_size: int,
    num_workers: int,
):
    """
    Creates PyTorch DataLoader objects for training and validation datasets from preprocessed data.
    Args:
        processed_data_dir (str | Path): Path to the directory containing the processed data.
        task_type (str): The type of task (MI or SSVEP) for which the data is prepared.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoader objects.
    """

    train, val = load_processed_data(
        processed_data_dir=processed_data_dir, task_type=task_type
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader


def setup_callbacks(
    model_config: ModelConfig,
    patience: int = 10,
    checkpoints_path: Path | None = None,
    verbose: bool = False,
):
    # Early stopping to prevent overfitting
    callbacks: List[Callback] = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            verbose=verbose,
        ),
    ]

    # Only add the ModelCheckpoint callback if a path is provided
    if checkpoints_path:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoints_path,
            monitor="val_f1",
            mode="max",  # Higher F1 is better
            save_top_k=3,  # Keep top 3 models
            filename=f"{model_config.name.lower()}-{model_config.task_type.lower()}-best-f1-{{val_f1:.4f}}-{{epoch:02d}}",
            save_last=True,  # Always save the last checkpoint
            verbose=verbose,
        )
        callbacks.append(checkpoint_callback)
    else:
        print("No checkpoints_path provided. Checkpointing is disabled.")

    return callbacks


def train_model(
    model: nn.Module,
    config_path: Path,
    checkpoints_path: Path | None = None,
    verbose: bool = True,
) -> Tuple[Trainer, BCILightningModule]:
    model_config = load_model_config(config_path)
    training_config = load_training_config(config_path)

    max_num_workers = rec_cpu_count()

    # Create data loaders
    train_loader, val_loader = create_processed_data_loaders(
        processed_data_dir=PROCESSED_DATA_DIR,
        task_type=model_config.task_type,
        batch_size=training_config.batch_size,
        num_workers=max_num_workers,
    )

    # Create Lightning module
    module = BCILightningModule(
        model=model,
        model_config=model_config,
        training_config=training_config,
    )

    # Setup callbacks
    callbacks = setup_callbacks(
        model_config,
        patience=training_config.patience,
        checkpoints_path=checkpoints_path,
        verbose=verbose,
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=training_config.epochs,
        callbacks=callbacks,
        accelerator="auto",  # Automatically uses GPU if available
        devices="auto",  # Uses all available devices
        deterministic=True,  # For reproducibility
        log_every_n_steps=10,
    )

    trainer.fit(module, train_loader, val_loader)

    return trainer, module


def train_and_save(
    task_type: str,
):
    config_path = None
    save_path = None
    if task_type.upper() == "MI":
        config_path = MI_CONFIG_PATH
        save_path = MI_RUNS_DIR
    elif task_type.upper() == "SSVEP":
        config_path = SSVEP_CONFIG_PATH
        save_path = SSVEP_RUNS_DIR
    else:
        raise (
            ValueError(
                f"Invalid task_type: {task_type}.\nValid task_type (MI) or (SSVEP)"
            )
        )

    model_config = load_model_config(config_path)

    # model to use
    model = get_model_class(model_config.name)
    model_name = model.__name__

    # Create a unique temporary directory first, using only the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_run_folder_name = f"{model_name}-{timestamp}-inprogress"
    temp_run_save_dir = save_path / temp_run_folder_name

    # Ensure directory and a subdirectory for checkpoints exist
    checkpoints_subdir = temp_run_save_dir / "checkpoints"
    os.makedirs(checkpoints_subdir, exist_ok=True)
    print(f"Created temporary run directory: {temp_run_save_dir}")

    model_kwargs = {
        "num_electrodes": model_config.num_channels,
        "chunk_size": model_config.new_sequence_length,
        "num_classes": model_config.num_classes,
        **model_config.params,
    }
    model = model(**model_kwargs)

    trainer, model = train_model(
        model=model,
        config_path=config_path,
        checkpoints_path=checkpoints_subdir,
        verbose=True,
    )

    f1_score = trainer.callback_metrics.get("val_f1")
    if f1_score is None:
        # Fallback for early failure
        f1_score = 0.0

    # Create the final directory name with f1 score
    final_run_folder_name = f"{model_name}-f1-{f1_score:.4f}-{timestamp}"
    final_save_dir = save_path / final_run_folder_name

    # Rename the temporary directory to its final name
    os.rename(temp_run_save_dir, final_save_dir)
    print(f"Renamed run directory to: {final_save_dir}")

    # Copy config file to new model directory
    shutil.copy(config_path, final_save_dir / "config.yaml")
    print(f"Saved config to {final_save_dir / 'config.yaml'}")

    save_model(model=model, save_path=final_save_dir / "weights.pt")
    print(f"Saved config to {final_save_dir / 'weights.pt'}")


def main():
    # To utilize cuda cores
    torch.set_float32_matmul_precision("medium")

    # Code necessary to create reproducible runs
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Argument parser for cli use
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        required=True,
        help="Task type (MI or SSVEP).",
    )
    args = parser.parse_args()

    train_and_save(
        task_type=args.task_type,
    )


if __name__ == "__main__":
    main()
