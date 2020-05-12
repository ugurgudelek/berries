__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"
import torch
from torch import nn

class CNN1D(nn.Module):

    def __init__(self):
        super(CNN1D, self).__init__()

        self.features = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=10,
                                                kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True),
                                      nn.MaxPool1d(kernel_size=4, stride=2, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.Conv1d(in_channels=10, out_channels=10,
                                                kernel_size=5, stride=1, padding=0, dilation=1, groups=1, bias=True),
                                      nn.MaxPool1d(kernel_size=4, stride=2, padding=0),
                                      nn.ReLU(inplace=True)
                                      )

        self.classifier = nn.Sequential(nn.Linear(in_features=7080, out_features=256),
                                        nn.Dropout(p=0.90, inplace=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=256, out_features=128),
                                        nn.Dropout(p=0.90, inplace=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=128, out_features=2))

    def forward(self, x):
        x = x.view(-1, 1, x.shape[2] * x.shape[3])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN2D(nn.Module):

    def __init__(self):
        super(CNN2D, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10,
                                                kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=4, stride=2, padding=0),
                                      nn.Conv2d(in_channels=10, out_channels=10,
                                                kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=4, stride=2, padding=0))

        self.classifier = nn.Sequential(nn.Linear(in_features=990, out_features=256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=256, out_features=128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=128, out_features=3))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class NN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.6):
        super(NN, self).__init__()

        self.fc = nn.Sequential(nn.Linear(in_features=input_dim, out_features=200),
                                nn.Dropout(p=dropout, inplace=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=200, out_features=200),
                                nn.Dropout(p=dropout, inplace=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=200, out_features=100),
                                nn.Dropout(p=dropout, inplace=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=100, out_features=output_dim))


    def forward(self, x, **kwargs):
        if x.dim() > 2:
            x = x.view(-1, x.shape[2] * x.shape[3])
        x = self.fc(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()

        self.fc = nn.Linear(in_features=input_dim, out_features=output_dim)

        # self.fc = nn.Sequential(nn.Linear(in_features=5, out_features=3))

    def forward(self, x, **kwargs):
        if x.dim() > 2:
            x = x.view(-1, x.shape[2] * x.shape[3])
        x = self.fc(x)
        return x