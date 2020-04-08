__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from torch import nn


class CNN(nn.Module):
    """Basic Pytorch CNN implementation"""

    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=(3, 6), stride=(1,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 6)))
        self.c2 =nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(3, 6), stride=(1,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 6)))
        self.c3 =nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=10, kernel_size=(3, 6), stride=(1,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 6)))


        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=100),
            nn.Linear(in_features=100, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)

        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = out.view(batch_size, -1) # flatten the vector
        out = self.classifier(out)
        return out
