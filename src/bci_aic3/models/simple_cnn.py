# models/simple_cnn.py
import torch
from torch import nn


# Simple CNN model
class BCIModel(nn.Module):
    def __init__(self, input_channels, num_classes, sequence_length):
        super(BCIModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3)  # 8 channels
        self.pool = nn.MaxPool1d(2)

        # Helper to calculate the flattened size
        self._get_conv_output_size(input_channels, sequence_length)

        self.fc1 = nn.Linear(self._to_linear, num_classes)

    def _get_conv_output_size(self, input_channels, sequence_length):
        # Create a dummy input tensor
        dummy_input = torch.randn(1, input_channels, sequence_length)

        # Pass it through the convolutional and pooling layers
        output = self.pool(torch.relu(self.conv1(dummy_input)))

        # Calculate the flattened size
        self._to_linear = output.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
