__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn


class LSTM(nn.Module):
    """
    Simplest LSTM Implementation for time series.
    Keep in mind that after each forward pass, hiddens are detached from computational
    network but they keep their computed values.
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1,
                 num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.hidden = None

    def forward(self, input):
        """

        :param input (batch_size, seq_len, input_dim):
        :return:
        """

        # Inputs: input, (h_0, c_0)
        # input: (seq_len, batch, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        # c_0: (num_layers * num_directions, batch, hidden_size)

        batch, seq_len, input_size = input.shape
        lstm_out, self.hidden = self.lstm(input.view(seq_len, batch, input_size), self.hidden)
        # Outputs: output,
        # output: (seq_len, batch, num_directions * hidden_size)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)

        self.hidden = self.hidden[0].detach(), self.hidden[1].detach()

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.linear(lstm_out[-1, :, :])
        # return y_pred.view(-1)

        y_pred = self.linear(lstm_out)
        return y_pred.view(batch, seq_len)



