import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from keras.utils import to_categorical

# First: we need to read the text into io and encode the text with integers.

# read and prepare the data
with open('../anna.txt', 'r') as f:
    text = f.read()

# get the set of all characters
characters = tuple(set(text))

# use enumeration to give the characters integer values
int2char = dict(enumerate(characters))

# create the look up dictionary from characters to the assigned integers
char2int = {char: index for index, char in int2char.items()}

# encode the text, using the character to integer dictionary
encoded = np.array([char2int[char] for char in text])

#Second: we need to write the batching algorithm. As I mentioned before, we want to set the targets to be the training characters but shifted by 1 in time to define the sequential order. I used the batching function from one of the classes I took before at Udacity. In my opinion, writing the batching algorithm is the hardest part for this type of task.

# batching function
def get_batches(arr, n_seqs_in_a_batch, n_characters):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''

    batch_size = n_seqs_in_a_batch * n_characters
    n_batches = len(arr) // batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs_in_a_batch, -1))

    for n in range(0, arr.shape[1], n_characters):
        # The features
        x = arr[:, n:n + n_characters]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_characters]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

# Third: we want to start building our model in pytorch. Here we are going to use the LSTM cell class to define the cells for both layers in our LSTM model. It is common to initialize the hidden and cell states to tensors of zeros to pass to the first LSTM cell in the sequence.
# build the model using the pytorch nn module
class CharLSTM(nn.ModuleList):
    def __init__(self, sequence_len, vocab_size, hidden_dim, batch_size):
        super(CharLSTM, self).__init__()

        # init the meta parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size

        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim)

        # second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # dropout layer for the output of the second layer cell
        self.dropout = nn.Dropout(p=0.5)

        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, hc):
        """
            x: input to the model
                *  x[t] - input of shape (batch, input_size) at time t

            hc: hidden and cell states
                *  tuple of hidden and cell state
        """

        # empty tensor for the output of the lstm
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.vocab_size))

        # pass the hidden and the cell state from one lstm cell to the next one
        # we also feed the output of the first layer lstm cell at time step t to the second layer cell
        # init the both layer cells with the zero hidden and zero cell states
        hc_1, hc_2 = hc, hc

        # for every step in the sequence
        for t in range(self.sequence_len):
            # get the hidden and cell states from the first layer cell
            hc_1 = self.lstm_1(x[t], hc_1)

            # unpack the hidden and the cell states from the first layer
            h_1, c_1 = hc_1

            # pass the hidden state from the first layer to the cell in the second layer
            hc_2 = self.lstm_2(h_1, hc_2)

            # unpack the hidden and cell states from the second layer cell
            h_2, c_2 = hc_2

            # form the output of the fc
            output_seq[t] = self.fc(self.dropout(h_2))

        # return the output sequence
        return output_seq.view((self.sequence_len * self.batch_size, -1))

    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.batch_size, self.hidden_dim).to(device))

    def init_hidden_predict(self, b):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(b, self.hidden_dim).to(device),
                torch.zeros(b, self.hidden_dim).to(device))

    def predict(self, char, top_k=5, seq_len=128):
        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        '''

        # set the evaluation mode
        self.eval()

        # placeholder for the generated text
        seq = np.empty(seq_len + 1)
        seq[0] = char2int[char]

        # initialize the hidden and cell states
        hc = self.init_hidden_predict(1)

        # now we need to encode the character - (1, vocab_size)
        char = to_categorical(char2int[char], num_classes=self.vocab_size)

        # add the batch dimension
        char = torch.from_numpy(char).unsqueeze(0).to(device)

        # now we need to pass the character to the first LSTM cell to obtain
        # the predictions on the second character
        hc_1, hc_2 = hc, hc

        # for the sequence length
        for t in range(seq_len):
            # get the hidden and cell states from the first LSTM layer
            hc_1 = self.lstm_1(char, hc_1)
            h_1, _ = hc_1

            # get the hidden and cell states from the second LSTM layer
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, _ = hc_2

            # pass the output of the cell through fully connected layer
            h_2 = self.fc(h_2)

            # apply the softmax to the output to get the probabilities of the characters
            h_2 = F.softmax(h_2, dim=1)

            # h_2 now holds the vector of predictions (1, vocab_size)
            # we want to sample 5 top characters
            p, top_char = h_2.topk(top_k)

            # get the top k characters by their probabilities
            top_char = top_char.squeeze().cpu().numpy()

            # sample a character using its probability
            p = p.detach().squeeze().cpu().numpy()
            char = np.random.choice(top_char, p=p / p.sum())

            # append the character to the output sequence
            seq[t + 1] = char

            # prepare the character to be fed to the next LSTM cell
            char = to_categorical(char, num_classes=self.vocab_size)
            char = torch.from_numpy(char).unsqueeze(0).to(device)

        return seq

# Forth: now we want to define our model object along with the optimizer and the loss function. We are going to use Adam optimizer since it is the most common choice for such tasks. Also, we are going to use the cross-entropy loss as we are going to measure entropy between our output and the targets (which are distributions).
# compile the network - sequence_len, vocab_size, hidden_dim, batch_size
cuda = torch.cuda.is_available()
device = 'cpu'
if cuda:
    device = 'cuda'
net = CharLSTM(sequence_len=128, vocab_size=len(char2int), hidden_dim=512, batch_size=128).to(device)

# define the loss and the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Fifth: we want to train our model. I am going to separate the encoded data-set into training and validation sets to monitor the losses to diagnose over of under fitting.
# get the validation and the training data
val_idx = int(len(encoded) * (1 - 0.1))
data, val_data = encoded[:val_idx], encoded[val_idx:]

# empty list for the validation losses
val_losses = list()

# empty list for the samples
samples = list()

for epoch in range(10):

    # reinit the hidden and cell steates
    hc = net.init_hidden()

    for i, (x, y) in enumerate(get_batches(data, 128, 128)):

        # get the torch tensors from the one-hot of training data
        # also transpose the axis for the training set and the targets
        x_train = torch.from_numpy(to_categorical(x, num_classes=net.vocab_size).transpose([1, 0, 2])).to(device)
        targets = torch.from_numpy(y.T).type(torch.LongTensor).to(device)  # tensor of the target

        # zero out the gradients
        optimizer.zero_grad()

        # get the output sequence from the input and the initial hidden and cell states
        output = net(x_train, hc).to(device)


        # calculate the loss
        # we need to calculate the loss across all batches, so we have to flat the targets tensor
        loss = criterion(output, targets.contiguous().view(128 * 128))

        # calculate the gradients
        loss.backward()

        # update the parameters of the model
        optimizer.step()

        # feedback every 10 batches
        if i % 10 == 0:

            # initialize the validation hidden state and cell state
            val_h, val_c = net.init_hidden()

            for val_x, val_y in get_batches(val_data, 128, 128):
                # prepare the validation inputs and targets
                val_x = torch.from_numpy(to_categorical(val_x).transpose([1, 0, 2])).to(device)
                val_y = torch.from_numpy(val_y.T).type(torch.LongTensor).contiguous().view(128 * 128).to(device)

                # get the validation output
                val_output = net(val_x, (val_h, val_c)).to(device)

                # get the validation loss
                val_loss = criterion(val_output, val_y)

                # append the validation loss
                val_losses.append(val_loss.item())

                # sample 256 chars
                samples.append(''.join([int2char[int_] for int_ in net.predict("A", seq_len=1024)]))

            print("Epoch: {}, Batch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}".format(epoch, i, loss.item(),
                                                                                             val_loss.item()))
            print(samples[-1])


#


