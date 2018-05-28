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


def get_model_cls_from_name(name):
    if name == 'CNN':
        return CNN

    if name == 'LSTM':
        return LSTM

class CNN(nn.Module):
    """

    """
    def __init__(self, model_name, input_size, out_size, batch_size):
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
            nn.Linear(500, out_size)
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


    def __init__(self, input_size, seq_length, num_layers, out_size, batch_size, use_cuda):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.out_size = out_size
        self.batch_size = batch_size

        self.use_cuda = use_cuda

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
                            hidden_size=10,
                            num_layers=self.num_layers,
                            bias=True,
                            batch_first=False,
                            dropout=0.2,
                            bidirectional=False)

        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=self.seq_length, out_features=100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=100, out_features=self.out_size)
        # )

        self.hidden = None



    def init_hidden(self, batch_size=None):
        """

        Returns:
            (Variable,Variable): (h_0, c_0)

        """
        if batch_size is None:
            batch_size = self.batch_size
        (h0,c0) = (Variable(torch.zeros(self.num_layers,batch_size, 10)),  # h_0
                Variable(torch.zeros(self.num_layers,batch_size, 10)))  # c_0

        if self.use_cuda:
            return h0.cuda(), c0.cuda()

        return h0, c0

    # def init_hidden_2(self, bsz):
    #     weight = next(self.parameters()).data
    #     a = weight.new(self.num_layers, bsz, 1).normal_(-1, 1)
    #     b = weight.new(self.num_layers, bsz, 10 - 1).zero_()
    #     return Variable(torch.cat([a, b], 2))

    def forward(self, x):

        # Reshape input
        # x shape: (seq_len, batch, input_size)
        # hidden shape:(num_layers * num_directions,batch_size, hidden_size)

        if self.hidden is None:
            self.hidden = self.init_hidden()

        x = x.view(self.seq_length, -1, self.input_size)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # return all sequence, all batches, only last hidden
        return lstm_out[:, :, -1].view(-1, self.seq_length)

# ## A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
# source: https://arxiv.org/pdf/1704.02971.pdf
# pytorch example: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, T, logger):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.logger = logger

        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
        self.attn_linear = nn.Linear(in_features = 2 * hidden_size + T - 1, out_features = 1)

    def forward(self, input_data):
        # input_data: batch_size * T - 1 * input_size
        input_weighted = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.input_size).zero_())
        input_encoded = Variable(input_data.data.new(input_data.size(0), self.T - 1, self.hidden_size).zero_())
        # hidden, cell: initial states with dimention hidden_size
        hidden = self.init_hidden(input_data) # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim = 2) # batch_size * input_size * (2*hidden_size + T - 1)
            # Eqn. 9: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1)) # (batch_size * input_size) * 1
            attn_weights = F.softmax(x.view(-1, self.input_size)) # batch_size * input_size, attn weights with values sum up to 1.
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :]) # batch_size * input_size
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
        return input_weighted, input_encoded

    def init_hidden(self, x):
        # No matter whether CUDA is used, the returned variable will have the same type as x.
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_()) # dimension 0 is the batch dimension

class decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T, logger):
        super(decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.logger = logger

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                         nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size = 1, hidden_size = decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T - 1 * encoder_hidden_size
        # y_history: batch_size * (T-1)
        # Initialize hidden and cell, 1 * batch_size * decoder_hidden_size
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        # hidden.requires_grad = False
        # cell.requires_grad = False
        for t in range(self.T - 1):
            # Eqn. 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2), input_encoded), dim = 2)
            x = F.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size
                                                )).view(-1, self.T - 1)) # batch_size * T - 1, row sum up to 1
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :] # batch_size * encoder_hidden_size
            if t < self.T - 1:
                # Eqn. 15
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim = 1)) # batch_size * 1
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0] # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1] # 1 * batch_size * decoder_hidden_size
        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim = 1))
        # self.logger.info("hidden %s context %s y_pred: %s", hidden[0][0][:10], context[0][:10], y_pred[:10])
        return y_pred

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.decoder_hidden_size).zero_())


# class LoadLSTM(nn.Module):
#     """
#         Long-short term memory implementation for LoadDataset
#
#         Args:
#             input_size:
#             seq_length:
#             num_layers:
#
#         Attributes:
#             input_size:
#             seq_length:
#             num_layers:
#             lstm:
#             hidden:
#     """
#
#     def __init__(self, input_size, seq_length, num_layers, batch_size):
#         super(LoadLSTM, self).__init__()
#         self.input_size = input_size
#         self.seq_length = seq_length
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#
#         # Inputs: input, (h_0,c_0)
#         #   input(seq_len, batch, input_size)
#         #   h_0(num_layers * num_directions, batch, hidden_size)
#         #   c_0(num_layers * num_directions, batch, hidden_size)
#         #   Note:If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
#         # Outputs: output, (h_n, c_n)
#         #   output (seq_len, batch, hidden_size * num_directions)
#         #   h_n (num_layers * num_directions, batch, hidden_size)
#         #   c_n (num_layers * num_directions, batch, hidden_size)
#         self.lstm = nn.LSTM(input_size=self.input_size,
#                             hidden_size=self.seq_length,
#                             num_layers=self.num_layers)
#
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         """
#
#         Returns:
#             (Variable,Variable): (h_0, c_0)
#
#         """
#         # (num_layers, batch, hidden_dim)
#         return (Variable(torch.zeros(self.num_layers, 1, self.seq_length)),  # h_0
#                 Variable(torch.zeros(self.num_layers, 1, self.seq_length)))  # c_0
#
#     def forward(self, x):
#         """
#
#         Args:
#             x:
#
#         Returns:
#             (?,?)
#         """
#         # Reshape input
#         # x shape: (seq_len, batch, input_size)
#         # hidden shape:(num_layers * num_directions,batch_size, hidden_size)
#         x = x.view(self.seq_length, -1, self.input_size)
#         lstm_out, self.hidden = self.lstm(x, self.hidden)
#         return lstm_out
