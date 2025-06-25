# sota_eeg_models.py
#
# This file contains PyTorch implementations of several state-of-the-art
# deep learning architectures for EEG-based Motor Imagery (MI) classification.
# The models included are:
# 1. EEGNet (from user's prompt, for reference)
# 2. DeepConvNet
# 3. ShallowConvNet
# 4. ATCNet (Attention-based Temporal Convolutional Network)
#
# All models are designed to accept inputs with a similar signature for
# consistency and ease of use in BCI pipelines.

import math
import torch
import torch.nn as nn

# ===================================================================================================
# 1. DeepConvNet (SOTA, from https://doi.org/10.1002/hbm.23730)
# ===================================================================================================


class DeepConvNet(nn.Module):
    """
    DeepConvNet model for EEG MI classification.
    A classic, robust architecture for various EEG tasks.
    Inputs:
        num_classes: int, number of classes to classify.
        channels: int, number of EEG channels.
        samples_per_trial: int, number of time points. (Not directly used, but good for consistency)
        dropoutRate: float, dropout probability.
    """

    def __init__(
        self, num_classes=4, channels=22, samples_per_trial=2250, dropoutRate=0.5
    ):
        super(DeepConvNet, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), bias=False),
            nn.Conv2d(25, 25, kernel_size=(channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=dropoutRate),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=dropoutRate),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=dropoutRate),
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=dropoutRate),
        )

        self.flatten = nn.Flatten()

        # Calculate feature size dynamically for the classifier
        dummy_input = torch.randn(1, 1, channels, samples_per_trial)
        feature_size = self._get_feature_size(dummy_input)

        self.classifier = nn.Linear(feature_size, num_classes)

    def _get_feature_size(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        # Input shape: (batch, channels, samples) -> (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# ===================================================================================================
# 2. ShallowConvNet (SOTA, from https://doi.org/10.1002/hbm.23730)
# ===================================================================================================
class ShallowConvNet(nn.Module):
    """
    ShallowConvNet model for EEG MI classification.
    A compact and effective architecture inspired by FBCSP.
    Inputs:
        num_classes: int, number of classes.
        channels: int, number of EEG channels.
        samples_per_trial: int, number of time points. (Not directly used)
        dropoutRate: float, dropout probability.
    """

    def __init__(
        self, num_classes=4, channels=22, samples_per_trial=2250, dropoutRate=0.5
    ):
        super(ShallowConvNet, self).__init__()

        self.conv_block = nn.Sequential(
            # Temporal Convolution
            nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False),
            # Spatial Convolution
            nn.Conv2d(40, 40, kernel_size=(channels, 1), bias=False),
            nn.BatchNorm2d(40),
            # Squaring non-linearity is applied in forward pass
        )

        # Pooling and activation
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        # Log activation is applied in forward pass

        self.dropout = nn.Dropout(p=dropoutRate)
        self.flatten = nn.Flatten()

        # Calculate feature size dynamically
        dummy_input = torch.randn(1, 1, channels, samples_per_trial)
        feature_size = self._get_feature_size(dummy_input)

        self.classifier = nn.Linear(feature_size, num_classes)

    def _get_feature_size(self, x):
        x = self.conv_block(x)
        # Activation: square
        x = x**2
        x = self.pool(x)
        # Activation: log
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        # Input shape: (batch, channels, samples) -> (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        # Apply squaring non-linearity
        x = x**2
        x = self.pool(x)
        # Apply logarithmic non-linearity
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# ===================================================================================================
# 3. ATCNet (SOTA, from https://doi.org/10.1109/TNSRE.2021.3075591)
# ===================================================================================================


class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout
    ):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection if dimensions differ
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual[:, :, :, : out.size(3)])


class ATCNet(nn.Module):
    """
    Attention-based Temporal Convolutional Network (ATCNet).
    Combines CNNs with attention and a TCN block.
    Inputs:
        num_classes: int, number of classes.
        channels: int, number of EEG channels.
        samples_per_trial: int, number of time points.
        n_heads: int, number of attention heads.
        tcn_layers: int, number of TCN blocks.
        tcn_kernel_size: int, kernel size for TCN.
        dropoutRate: float, dropout probability.
    """

    def __init__(
        self,
        num_classes=4,
        channels=22,
        samples_per_trial=2250,
        n_heads=4,
        tcn_layers=2,
        tcn_kernel_size=4,
        F1=16,
        D=2,
        dropoutRate=0.3,
    ):
        super(ATCNet, self).__init__()

        # Initial Convolutional Block (from EEGNet)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate),
        )

        # Attention Block (Multi-Head Self Attention on spatial dimension)
        self.attention = nn.MultiheadAttention(
            embed_dim=F1 * D, num_heads=n_heads, dropout=dropoutRate, batch_first=True
        )

        # TCN Block
        self.tcn_blocks = nn.ModuleList()
        for i in range(tcn_layers):
            dilation_size = 2**i
            in_channels_tcn = F1 * D
            out_channels_tcn = F1 * D
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels_tcn,
                    out_channels_tcn,
                    kernel_size=(1, tcn_kernel_size),
                    stride=1,
                    dilation=(1, dilation_size),
                    padding=(0, (tcn_kernel_size - 1) * dilation_size // 2),
                    dropout=dropoutRate,
                )
            )

        # Classifier
        self.flatten = nn.Flatten()
        dummy_input = torch.randn(1, 1, channels, samples_per_trial)
        feature_size = self._get_feature_size(dummy_input)
        self.classifier = nn.Linear(feature_size, num_classes)

    def _get_feature_size(self, x):
        x = self.conv_block(x)

        # Attention expects (batch, seq_len, embed_dim)
        # Reshape: (N, C, H, W) -> (N, W, C*H) for spatial attention
        # Here H=1, so it becomes (N, W, C)
        N, C, H, W = x.shape
        x = x.squeeze(2)  # Remove height dim -> (N, C, W)
        x = x.permute(0, 2, 1)  # -> (N, W, C) i.e. (batch, sequence, features)

        attn_output, _ = self.attention(x, x, x)

        # Reshape back for TCN: (N, W, C) -> (N, C, 1, W)
        x = attn_output.permute(0, 2, 1).unsqueeze(2)

        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        # Input shape: (batch, channels, samples) -> (batch, 1, channels, samples)
        x = x.unsqueeze(1)

        # Conv Block
        x = self.conv_block(x)  # (N, F1*D, 1, W_out)

        # Reshape for Attention
        N, C, H, W = x.shape
        x = x.squeeze(2)  # (N, C, W)
        x = x.permute(0, 2, 1)  # (N, W, C) - Batch, Sequence Length, Embedding Dim

        # Attention Block
        attn_output, _ = self.attention(x, x, x)  # (N, W, C)

        # Reshape for TCN
        x = attn_output.permute(0, 2, 1).unsqueeze(2)  # (N, C, 1, W)

        # TCN Block
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        # Classifier
        x = self.flatten(x)
        x = self.classifier(x)

        return x


# ===================================================================================================
# 4. SSVEPformer
# ===================================================================================================


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformer models.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)


class SSVEPformer(nn.Module):
    """
    A PyTorch implementation of a Transformer-based model for SSVEP classification.

    Inputs:
        num_classes: int, number of classes (SSVEP frequencies).
        channels: int, number of EEG channels.
        samples_per_trial: int, number of time points in a trial.
        d_model: int, the dimension of the embeddings.
        nhead: int, the number of heads in the multiheadattention models.
        d_hid: int, the dimension of the feedforward network model.
        nlayers: int, the number of sub-encoder-layers in the encoder.
        dropout: float, the dropout value.
    """

    def __init__(
        self,
        num_classes=4,
        channels=22,
        samples_per_trial=1125,
        d_model=64,
        nhead=8,
        d_hid=256,
        nlayers=2,
        dropout=0.5,
    ):
        super(SSVEPformer, self).__init__()

        # Patch embedding layer: Conv2d to segment the time series
        # We use a kernel size of (1, 16) and stride of (1, 8) to create overlapping patches
        self.patch_embedding = nn.Conv2d(
            1, d_model, kernel_size=(channels, 16), stride=(1, 8)
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Standard Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.d_model = d_model
        self.flatten = nn.Flatten()

        # Classifier
        # Calculate the feature size after the transformer
        dummy_input = torch.randn(1, 1, channels, samples_per_trial)
        feature_size = self._get_feature_size(dummy_input)
        self.classifier = nn.Linear(feature_size, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def _get_feature_size(self, x):
        # Pass through patch embedding
        x = self.patch_embedding(x)  # Shape: (batch, d_model, 1, num_patches)
        x = x.squeeze(2)  # Shape: (batch, d_model, num_patches)
        x = x.permute(0, 2, 1)  # Shape: (batch, num_patches, d_model)

        # Pass through transformer
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, src):
        """
        Input shape: (batch_size, channels, samples_per_trial)
        """
        # Add a singleton dimension for Conv2d
        src = src.unsqueeze(1)  # Shape: (batch, 1, channels, samples)

        # 1. Patch Embedding
        # The Conv2d layer acts as a patch creator and embedder
        src = self.patch_embedding(src)  # Shape: (batch, d_model, 1, num_patches)
        src = src.squeeze(2)  # Shape: (batch, d_model, num_patches)
        src = src.permute(0, 2, 1)  # Shape: (batch, num_patches, d_model)

        # 2. Positional Encoding (Only applied if batch_first=False in Transformer)
        # For batch_first=True, we add it directly. We'll permute, add, and permute back.
        # src = src.permute(1, 0, 2) # seq, batch, feature
        # src = self.pos_encoder(src)
        # src = src.permute(1, 0, 2) # batch, seq, feature

        # 3. Transformer Encoder
        output = self.transformer_encoder(src)

        # 4. Classifier
        output = self.flatten(output)
        output = self.classifier(output)

        return output
