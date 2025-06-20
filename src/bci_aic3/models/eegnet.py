# src/models/eegnet.py

# Original paper - https://arxiv.org/abs/1611.08024

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class EEGNet(nn.Module):
    """SSVEP Variant of EEGNet, as used in [1].

    Inputs:
        num_classes      : int, number of classes to classify
        channels, samples_per_trial  : number of channels and time points in the EEG data
        dropoutRate     : dropout fraction
        kernLength      : length of temporal convolution in first layer
        F1, F2          : number of temporal filters (F1) and number of pointwise
                          filters (F2) to learn.
        D               : number of spatial filters to learn within each temporal
                          convolution.
        dropoutType     : Either 'SpatialDropout2D' or 'Dropout', passed as a string.

    [1]. Waytowich, N. et. al. (2018). Compact Convolutional Neural Networks
    for Classification of Asynchronous Steady-State Visual Evoked Potentials.
    Journal of Neural Engineering vol. 15(6).
    http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8
    """

    def __init__(
        self,
        num_classes=4,
        channels=8,
        samples_per_trial=1750,
        dropoutRate=0.25,
        kernLength=250,
        F1=96,
        D=1,
        F2=96,
        dropoutType="Dropout",
    ):
        super(EEGNet, self).__init__()

        self.num_classes = num_classes
        self.channels = channels
        self.samples_per_trial = samples_per_trial
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Validate dropout type
        if dropoutType not in ["SpatialDropout2D", "Dropout"]:
            raise ValueError(
                "dropoutType must be one of SpatialDropout2D "
                "or Dropout, passed as a string."
            )
        self.dropoutType = dropoutType

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Depthwise convolution - equivalent to DepthwiseConv2D
        self.depthwise_conv = nn.Conv2d(
            F1, F1 * D, (channels, 1), groups=F1, bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)

        # Apply max_norm constraint to depthwise conv weights
        self.depthwise_conv = weight_norm(self.depthwise_conv, name="weight")

        self.avgpool1 = nn.AvgPool2d((1, 4))

        # Block 2
        # SeparableConv2D is equivalent to depthwise + pointwise convolution
        self.separable_conv_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), groups=F1 * D, padding="same", bias=False
        )
        self.separable_conv_pointwise = nn.Conv2d(F1 * D, F2, 1, bias=False)

        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))

        # Calculate the size after convolutions for the linear layer
        # We need to be more careful about the feature size calculation
        # This will be computed dynamically in the first forward pass
        self.feature_size = None

        # Dense layer - will be initialized on first forward pass
        self.classifier = None

        # Dropout layers
        if dropoutType == "SpatialDropout2D":
            self.dropout1 = SpatialDropout2d(dropoutRate)
            self.dropout2 = SpatialDropout2d(dropoutRate)
        else:  # 'Dropout'
            self.dropout1 = nn.Dropout2d(dropoutRate)
            self.dropout2 = nn.Dropout2d(dropoutRate)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights similar to Keras defaults"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle different input formats
        # if x.dim() == 3:
        #     # Input shape: (batch_size, samples, channels)
        #     # Reshape to (batch_size, channels, samples, 1) then to (batch_size, 1, channels, samples)
        #     x = x.permute(0, 2, 1)  # (batch_size, channels, samples)
        #     x = x.unsqueeze(1)  # (batch_size, 1, channels, samples)
        # elif x.dim() == 4:
        #     # Input shape: (batch_size, channels, samples_per_trial, 1)
        #     # Reshape to (batch_size, 1, channels, samples_per_trial) for PyTorch conv2d
        #     x = x.permute(0, 3, 1, 2)
        # else:
        #     raise ValueError(f"Expected input to be 3D or 4D, got {x.dim()}D tensor")

        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separable_conv_depthwise(x)
        x = self.separable_conv_pointwise(x)
        x = self.batchnorm3(x)
        x = F.elu(x)

        # Check if we can apply the second pooling
        if x.size(-1) >= 8:  # Check if temporal dimension is >= 8
            x = self.avgpool2(x)
        else:
            # Use adaptive pooling or smaller kernel
            x = F.avg_pool2d(x, (1, min(x.size(-1), 2)))

        x = self.dropout2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Initialize classifier on first forward pass
        if self.classifier is None:
            self.feature_size = x.size(1)
            self.classifier = nn.Linear(self.feature_size, self.num_classes).to(
                x.device
            )
            # Initialize the classifier weights
            nn.init.kaiming_normal_(
                self.classifier.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.constant_(self.classifier.bias, 0)

        # Dense layer
        x = self.classifier(x)

        # Softmax (often applied in loss function, but included here for completeness)
        x = F.softmax(x, dim=1)

        return x


class SpatialDropout2d(nn.Module):
    """Spatial Dropout implementation for PyTorch

    Drops entire feature maps instead of individual elements.
    This is equivalent to Keras' SpatialDropout2D.
    """

    def __init__(self, p=0.5):
        super(SpatialDropout2d, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        # x shape: (N, C, H, W)
        N, C, H, W = x.size()

        # Create mask for entire feature maps
        mask = torch.bernoulli(torch.full((N, C, 1, 1), 1 - self.p, device=x.device))
        mask = mask.expand_as(x)

        # Apply mask and scale
        return x * mask / (1 - self.p)
