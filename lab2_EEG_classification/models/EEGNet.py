"""EEGNet and DeepConvNet for two-channel, 750-sample EEG."""
import math
from torch import nn


def activation_layer(name, alpha):
    if name == "elu":
        return nn.ELU(alpha=alpha)
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unknown activation: {name}")


def validate_options(alpha, dropout):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ELU alpha must be finite and positive")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("Dropout must be in [0, 1)")


class EEGNet(nn.Module):
    def __init__(self, activation="elu", alpha=1.0, dropout=0.25):
        super().__init__()
        validate_options(alpha, dropout)
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), groups=16, bias=False),
            nn.BatchNorm2d(32), activation_layer(activation, alpha),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32), activation_layer(activation, alpha),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.classify = nn.Linear(32 * 23, 2)

    def forward(self, x):
        x = self.separableConv(self.depthwiseConv(self.firstconv(x)))
        return self.classify(x.flatten(1))


class DeepConvNet(nn.Module):
    def __init__(self, activation="elu", alpha=1.0, dropout=0.5):
        super().__init__()
        validate_options(alpha, dropout)
        layers = [nn.Conv2d(1, 25, (1, 5)), nn.Conv2d(25, 25, (2, 1))]
        for i, channels in enumerate((25, 50, 100, 200)):
            if i:
                layers.append(nn.Conv2d(channels // 2, channels, (1, 5)))
            layers.extend([
                nn.BatchNorm2d(channels), activation_layer(activation, alpha),
                nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
            ])
        self.features = nn.Sequential(*layers)
        self.classify = nn.Linear(200 * 43, 2)

    def forward(self, x):
        return self.classify(self.features(x).flatten(1))
