# src/train.py

import argparse
import os
from pathlib import Path

import torch
import torchmetrics
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader

from bci_aic3.config import (
    ModelConfig,
    TrainingConfig,
    load_model_config,
    load_training_config,
)
from bci_aic3.data import load_processed_data, load_raw_data
from bci_aic3.models.eegnet import EEGNet
from bci_aic3.paths import (
    CHECKPOINTS_DIR,
    CONFIG_DIR,
    LABEL_MAPPING_PATH,
    MI_CONFIG_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SSVEP_CONFIG_PATH,
)
from bci_aic3.util import read_json_to_dict, rec_cpu_count


class BCILightningModule(LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        # TODO: Add more hyperparameters from config
        # optimizer_name: str = "adam",
        # weight_decay: float = 0.0,
        # scheduler_name: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves all __init__ params

        self.model_config = model_config
        self.training_config = training_config

        # Model
        self.model = EEGNet(
            self.model_config.num_classes,
            self.model_config.num_channels,
            self.model_config.sequence_length,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics - Lightning handles device placement automatically
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.model_config.num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.model_config.num_classes
        )

        self.train_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.model_config.num_classes,
            average="macro",
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.model_config.num_classes,
            average="macro",
        )

    def forward(self, x):
        # Lightning calls this for inference
        return self.model(x)  # Your transpose logic

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

    def configure_optimizers(self):
        # TODO: Make this configurable via hyperparameters
        optimizer = optim.Adam(self.parameters(), lr=self.training_config.learning_rate)

        # TODO: Add learning rate scheduler if needed
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",
        #     },
        # }

        return optimizer

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
    normalize: bool = True,
):
    """
    Creates PyTorch DataLoader objects for training and validation datasets from preprocessed data.
    Args:
        processed_data_dir (str | Path): Path to the directory containing the processed data.
        task_type (str): The type of task (MI or SSVEP) for which the data is prepared.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoader objects.
    """

    train, val = load_processed_data(
        processed_data_dir=processed_data_dir, task_type=task_type, normalize=normalize
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


def setup_callbacks(model_config):
    callbacks = [
        # Save best model based on F1 score (better for imbalanced classes)
        ModelCheckpoint(
            dirpath=CHECKPOINTS_DIR / model_config.task_type,
            monitor="val_f1",
            mode="max",  # Higher F1 is better
            save_top_k=3,  # Keep top 3 models
            filename=f"{model_config.name.lower()}-{model_config.task_type.lower()}-best-f1-{{val_f1:.4f}}-{{epoch:02d}}",
            save_last=True,  # Always save the last checkpoint
            verbose=True,
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=True,
        ),
    ]
    return callbacks


def main():
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

    config_path = None
    if args.task_type.lower() == "mi":
        config_path = MI_CONFIG_PATH
    elif args.task_type.lower() == "ssvep":
        config_path = SSVEP_CONFIG_PATH

    model_config = load_model_config(config_path)
    training_config = load_training_config(config_path)

    max_num_workers = rec_cpu_count()

    # Create data loaders
    train_loader, val_loader = create_processed_data_loaders(
        processed_data_dir=PROCESSED_DATA_DIR,
        task_type=model_config.task_type,
        batch_size=training_config.batch_size,
        num_workers=max_num_workers,
        normalize=True,
    )
    print("Loaded the data...")

    # Create Lightning module
    model = BCILightningModule(
        model_config=model_config,
        training_config=training_config,
    )

    # Setup callbacks
    callbacks = setup_callbacks(model_config)

    # Create trainer
    trainer = Trainer(
        max_epochs=training_config.epochs,
        callbacks=callbacks,
        accelerator="auto",  # Automatically uses GPU if available
        devices="auto",  # Uses all available devices
        deterministic=True,  # For reproducibility
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # TODO: Save final model in custom format if needed
    # best_model_path = trainer.checkpoint_callback.best_model_path
    # model = BCILightningModule.load_from_checkpoint(best_model_path)
    # save_model(model.model, MODEL_DIR / "")

    # print("Training completed!")
    # print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
