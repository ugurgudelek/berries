__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn

from berries.model.base import BaseModel


class LSTM(BaseModel):
    """
    Simplest LSTM Implementation for time series.
    Keep in mind that after each forward pass, hiddens are detached from computational
    network but they keep their computed values.
    """

    def __init__(self,
                 sequence_input_size,
                 hidden_size,
                 output_size=1,
                 num_layers=2,
                 batch_size=64,
                 scalar_input_size=0,
                 bidirectional=False,
                 stateful=False,
                 hidden_reset_period=None,
                 return_sequences=False):
        super(LSTM, self).__init__()
        self.sequence_input_size = sequence_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size  # todo: check this
        self.scalar_input_size = scalar_input_size
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences

        self.num_directions = 2 if self.bidirectional else 1

        self.stateful = stateful  # todo: check this
        self.step = 0
        if self.stateful and hidden_reset_period is None:
            raise Exception(
                "When stateful is True, you should pass hidden_reset_period")
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
        self.lstm = nn.LSTM(
            input_size=self.sequence_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            dropout=0.,  # default:0 means no probability
            bidirectional=self.bidirectional,
            batch_first=True)

        # Define the output layer
        in_features = self.hidden_size * self.num_directions + self.scalar_input_size
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features=in_features // 2,
                      out_features=in_features // 4),
            nn.ReLU(),
            nn.Linear(
                in_features=in_features // 4,
                out_features=self.output_size,
                # nn.Sigmoid(),   # ! be careful. with this setting, output cannot exceed 1.
            ))

        self.hidden = self._init_hidden()
        self._init_weights()

    @property
    def device(self):
        return torch.device('cuda') if next(
            self.parameters()).is_cuda else torch.device('cpu')

    def _init_weights(self):
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

    def _init_hidden(self, batch_size=None):
        """
        Returns:
            (Variable,Variable): (h_0, c_0)
            (num_layers * num_directions, batch, hidden_size)
        """
        if batch_size is None:
            batch_size = self.batch_size

        h0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(self.num_layers * self.num_directions, batch_size,
                         self.hidden_size).float()),
                          requires_grad=False).to(self.device)

        c0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.Tensor(self.num_layers * self.num_directions, batch_size,
                         self.hidden_size).float()),
                          requires_grad=False).to(self.device)

        return h0, c0

    def reset_states(self, batch_size=None):
        # print(f"States reset on {self.step}")
        self.hidden = self._init_hidden(batch_size=batch_size)

    def forward(self, x):
        """
        # required x shape: (batch_size, seq_len, sequence_input_size) because batch_first=True
        # required hidden shape:(num_layers * num_directions, batch_size, hidden_size)

        # hidden reset -> if not applied: the LSTM will treat a new batch as a continuation of a sequence (Stateful LSTM)
        #              -> if applied: Stateless LSTM
        # See. [Stateful vs Stateless LSTM](http://philipperemy.github.io/keras-stateful-lstm/)
        """

        sequence = x['sequences']
        scalars = x.get('scalars', None)

        batch_size, seq_len, sequence_input_size = sequence.shape

        if not self.stateful:
            self.reset_states(batch_size=batch_size)
        else:  # do not reset on every forward pass
            if (self.hidden_reset_period != -1) and (
                (self.step % self.hidden_reset_period) == 0):
                self.reset_states(batch_size=batch_size)
            self.step += 1  # increment step to determine to reset hidden

        # forward pass
        # auto iterates over seq_len
        lstm_out, self.hidden = self.lstm(sequence, self.hidden)
        # lstm_out'shape (batch, seq_len, num_directions * hidden_size)
        last_seq_out = lstm_out[:, -1, :]  # all_batch, last_seq, all_hidden

        if self.return_sequences:
            raise NotImplementedError()
            fc_in = lstm_out
            # fc_out shape: [batch_size, seq_len, output_size]

        else:
            fc_in = last_seq_out
            if scalars is not None:  # concat aux data
                fc_in = torch.cat((last_seq_out, scalars), dim=1)
            # tensor containing the output features (h_t) from the last layer of the LSTM

        fc_out = self.classifier(fc_in)

        # detach hidden, otherwise retain_graph=True is necessary if stateful=True.
        self.hidden = (self.hidden[0].detach_(), self.hidden[1].detach_())

        return fc_out
