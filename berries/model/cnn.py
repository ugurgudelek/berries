__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn

import functools
import operator


class CNN(nn.Module):
    """Basic Pytorch CNN implementation"""

    def __init__(self, in_channels, out_channels, input_dim):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=20,
                      kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=20, out_channels=50,
                      kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        num_features_before_fcnn = functools.reduce(operator.mul, list(
            self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=100),
            nn.Linear(in_features=100, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)

        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class ChatterCNN(nn.Module):
    """Chatter Pytorch CNN implementation"""

    def __init__(self, in_channels, out_channels, input_dim):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=20,
                      kernel_size=(2, 4), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)),

            nn.Conv2d(in_channels=20, out_channels=50,
                      kernel_size=(2, 4), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)),

            nn.Conv2d(in_channels=50, out_channels=10,
                      kernel_size=(2, 4), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)))

        num_features_before_fcnn = functools.reduce(operator.mul, list(
            self.feature_extractor(torch.rand(1, *input_dim)).shape))
        print(num_features_before_fcnn)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=100),
            nn.Linear(in_features=100, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)

        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out
