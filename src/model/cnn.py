__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from torch import nn


class CNN(nn.Module):
    """Basic Pytorch CNN implementation"""

    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=50 * 4 * 4, out_features=100),
            nn.Linear(in_features=100, out_features=out_channels),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(-1, 50 * 4 * 4)
        out = self.classifier(out)
        return out
