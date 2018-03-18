"""
Ugur Gudelek
model
ugurgudelek
06-Mar-18
finance-cnn
"""

import torch
from torch import nn
from torch.autograd import Variable

import torch.nn.functional as F

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


class LSTM(nn.Module):
    """

    """
    def __init__(self, input_size, seq_length, num_layers, out_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.out_size = out_size

        # Inputs: input, (h_0,c_0)
        #   input(seq_len, batch, input_size)
        #   h_0(num_layers * num_directions, batch, hidden_size)
        #   c_0(num_layers * num_directions, batch, hidden_size)
        #   Note:If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # Outputs: output, (h_n, c_n)
        #   output (seq_len, batch, hidden_size * num_directions)
        #   h_n (num_layers * num_directions, batch, hidden_size)
        #   c_n (num_layers * num_directions, batch, hidden_size)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.seq_length,
                            num_layers=self.num_layers,
                            bias=True,
                            batch_first=False,
                            dropout=0.2,
                            bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=10),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=10, out_features=self.out_size)
        )

        self.hidden = self.init_hidden()
    def init_hidden(self):
        """

        Returns:
            (Variable,Variable): (h_0, c_0)

        """
        return (Variable(torch.zeros(self.num_layers, 1, self.seq_length)),  # h_0
                Variable(torch.zeros(self.num_layers, 1, self.seq_length)))  # c_0

    def forward(self, x):

        # Reshape input
        # x shape: (seq_len, batch, input_size)
        # hidden shape:(num_layers * num_directions,batch_size, hidden_size)
        x = x.view(self.seq_length, -1, self.input_size)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        fc_out = self.fc(lstm_out[-1])

        # output = fc_out.data.numpy().argmax(axis=1)
        return fc_out