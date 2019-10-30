__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn

class LSTM(nn.Module):

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
        y_pred = self.linear(lstm_out[-1, :, :])
        return y_pred.view(-1)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):  # batch, seq_len , 1
        embeds = self.word_embeddings(sentence)  # batch, seq_len, embed_dim
        # input:  seq_len, batch_size, input_size
        # output: seq_len, batch, num_directions * hidden_size
        lstm_out, _ = self.lstm(embeds.view(len(sentence[0]), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(1, len(sentence[0]), -1))
        return tag_space