__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):
    """
    Simplest LSTM Implementation for time series.
    Keep in mind that after each forward pass, hiddens are detached from computational
    network but they keep their computed values.
    """

    def __init__(self, input_size, hidden_size, output_size=1,
                 num_layers=2, batch_size=64, stateful=False, hidden_reset_period=None):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size # todo: check this
        self.device = torch.device('cuda:0')

        self.stateful = stateful  # todo: check this
        self.step = 0
        if self.stateful and hidden_reset_period is None:
            raise Exception("When stateful is True, you should pass hidden_reset_period")
        self.hidden_reset_period = hidden_reset_period



        # Define the LSTM layer
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
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=False,
                            dropout=0.,  # default:0 means no probability
                            bidirectional=False,
                            batch_first=True
                            )

        # Define the output layer
        self.fc = nn.Sequential(nn.Linear(in_features=self.hidden_size, out_features=self.output_size))

        self.hidden = self.init_hidden()
        self.init_weights()

    def init_weights(self):
        # default lstm cell init
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
        """

        # default fc init
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        """
        pass

    def init_hidden(self, batch_size=None):
        """
        Returns:
            (Variable,Variable): (h_0, c_0)
            (num_layers * num_directions, batch, hidden_size)
        """
        if batch_size is None:
            batch_size = self.batch_size
        (h0, c0) = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),  # h_0
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))  # c_0

        return (h0.to(self.device), c0.to(self.device))

    def reset_states(self, batch_size=None):
        # print(f"States reset on {self.step}")
        self.hidden = self.init_hidden(batch_size=batch_size)

    def forward(self, x, return_sequences=False):
        """
        # required x shape: (batch_size, seq_len, input_size) because batch_first=True
        # required hidden shape:(num_layers * num_directions, batch_size, hidden_size)

        # hidden reset -> if not applied: the LSTM will treat a new batch as a continuation of a sequence (Stateful LSTM)
        #              -> if applied: Stateless LSTM
        # See. [Stateful vs Stateless LSTM](http://philipperemy.github.io/keras-stateful-lstm/)
        """

        # Inputs: input, (h_0, c_0)
        # input: (seq_len, batch, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        # c_0: (num_layers * num_directions, batch, hidden_size)

        seq_len, batch_size, input_size = x.shape

        if not self.stateful:
            self.reset_states(batch_size=batch_size)
        else:  # do not reset on every forward pass
            if (self.hidden_reset_period != -1) and ((self.step % self.hidden_reset_period) == 0):
                self.hidden = self._init_hidden(batch_size=batch_size)
            self.step += 1  # increment step to determine to reset hidden

        # forward pass
        # auto iterates over seq_len
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # return last sequence
        # lstm_out'shape (batch, seq_len, num_directions * hidden_size)
        # tensor containing the output features (h_t) from the last layer of the LSTM
        last_seq_out = lstm_out[:, -1, :]  # all_batch, last_seq, all_hidden
        last_hidden_out = lstm_out[:, :, -1]  # all_batch, last_seq, all_hidden

        if return_sequences:
            return last_seq_out

        # fc_seq_out = self.fc_seq(last_hidden_out)
        # fc_hidden_out = self.fc_hidden(last_seq_out)

        # fc_out = self.fc(torch.cat((fc_seq_out, fc_hidden_out), dim=0).view(1, -1))

        fc_out = self.fc(last_seq_out)

        # detach hidden, otherwise retain_graph=True is necessary if stateful=True.
        self.hidden = (self.hidden[0].detach_(), self.hidden[1].detach_())

        return fc_out



