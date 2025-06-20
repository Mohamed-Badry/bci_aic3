# src/train.py

import torch
import os
from pytorch_lightning import seed_everything
from torch import nn, optim
from torch.utils.data import DataLoader

from bci_aic3.data import load_data
from bci_aic3.models.simple_cnn import BCIModel
from bci_aic3.util import read_json_to_dict, rec_cpu_count, save_model
from bci_aic3.paths import RAW_DATA_DIR, LABEL_MAPPING_PATH


# Code necessary to create reproducible runs
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
seed_everything(42, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    task_type = "MI"  # MI or SSVEP
    label_mapping = read_json_to_dict(LABEL_MAPPING_PATH)
    sequence_length = None

    if task_type == "MI":
        num_classes = 2
        sequence_length = 2250
    elif task_type == "SSVEP":
        num_classes = 4
        sequence_length = 1750

    batch_size = 32
    max_num_workers = rec_cpu_count()

    # Loading the data
    train, val, _ = load_data(
        base_path=RAW_DATA_DIR, task_type=task_type, label_mapping=label_mapping
    )

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=max_num_workers
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=max_num_workers
    )

    # Defining the model, loss, and optimizer
    model = BCIModel(
        train[0][0].shape[1], num_classes=num_classes, sequence_length=sequence_length
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data = data.to(device)
            labels = data.to(device)

            optimizer.zero_grad()
            outputs = model(data.transpose(1, 2))

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = data.to(device)

                outputs = model(data.transpose(1, 2))

                val_loss += criterion(outputs, labels).item()

        print(f"Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader)}")


if __name__ == "__main__":
    main()
