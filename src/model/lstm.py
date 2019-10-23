__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from torch import nn

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