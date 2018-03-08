"""
Ugur Gudelek
model
ugurgudelek
06-Mar-18
finance-cnn
"""

import torch
from torch import nn
class CNN(nn.Module):
    """

    """
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1,4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=0.2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(4608, 500),
            nn.Linear(500, 1)
        )

    def forward(self, x):
        output = self.cnn1(x)
        output = self.cnn2(output)
        output = output.view(output.size()[0], -1) # flatten
        output = self.fc(output)
        return output



def main():
    pass


if __name__ == "__main__":
    main()