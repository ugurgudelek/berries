import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from functools import partial
from torch import optim
from tensorboardX import SummaryWriter

import config

class GenericModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)



        # ============== FIT - PREDICT - VALIDATE METHODS ====================
        self.train_on_batch = partial(self._on_batch, train=True)
        self.validate_on_batch = partial(self._on_batch, train=False)


        # Predict given xs at once
        self.predict = partial(self._on_batch, ys=None, train=False)

        # Fit model for given dataloader xs and ys
        self.fit = partial(self._on_dataloader, train=True)
        self.validate = partial(self._on_dataloader, train=False)

    def _on_dataloader(self, dataloader, train):
        """Never call this function directly!"""

        losses = np.array([])
        for step, (xs, ys) in enumerate(dataloader):
            xs = Variable(xs.float(), requires_grad=False)
            ys = Variable(ys.float(), requires_grad=False)

            if train:
                output,loss = self.train_on_batch(xs, ys)
            else:
                output,loss = self.validate_on_batch(xs, ys)

            losses = np.append(losses, loss.item())

        return losses.mean()

    def _on_batch(self, xs, ys, train):
        """Never call this function directly!"""
        # if this call for validation or test, change model mode to evaluation
        # this is necessary because dropout and batch normalization should behave differently on evaluation mode
        if not train:
            self.eval()

        self.optimizer.zero_grad()  # pytorch accumulates gradients.

        xs = xs.to(self.device)
        # forward
        output = self.forward(xs)
        loss = None

        if ys is not None:
            ys = ys.to(self.device)

            # loss
            if isinstance(self.criterion, nn.NLLLoss):
                ys = ys.long()
            loss = self.criterion(output, ys)

            if train:

                # backward
                loss.backward(retain_graph=True)
                #optimize
                self.optimizer.step()

        # detach first hiddens of previous iteration
        if isinstance(self, LSTM):
            self.detach()

        # if this call for validation or test, model mode has changed to eval before,
        # now we need to revert this behaviour
        if not train:
            self.train()

        return output, loss


    def to_onnx(self, directory):
        # todo: below line not working right now. so check:
        # https://github.com/lanpa/tensorboardX/issues/166
        # estimator.writer.add_graph(model, (dummy_input,))

        assert 'dummy_input' in dir(self), 'dummy_input method should be implemented in model class!'
        torch.onnx.export(self, self.dummy_input(), os.path.join(directory, 'model.onnx'), verbose=True)

    def to_txt(self, directory):
        with open(os.path.join(directory, 'model.txt'), 'w') as f:
            f.write(self.__str__())

    def get_layers(self):
        layers = []

        def _recursive_get_layers(network):
            for layer in network.children():
                if isinstance(layer,
                              nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                    _recursive_get_layers(layer)
                if list(layer.children()).__len__() == 0:  # if leaf node, add it to list
                    layers.append(layer)

        _recursive_get_layers(network=self)  # start with whole network

        return layers

    def weight_bias_name(self):
        "Generator for weight, bias, name"
        layers = self.get_layers()
        title = None
        weights = None
        for i, layer in enumerate(layers):
            # LSTM Layer
            if isinstance(layer, nn.LSTM):
                num_lstm_layer = layer.state_dict().keys().__len__()//4  # input,hidden,weight,bias for each

                for i_h in ['i', 'h']:
                    for layer_num in range(num_lstm_layer):
                        name = '{i_or_h}h_l{layer_num}'.format(i_or_h=i_h, layer_num=layer_num)
                        weight_key = 'weight_'+name
                        bias_key = 'bias_'+name

                        weight = layer.state_dict()[weight_key].data.numpy()
                        bias = layer.state_dict()[bias_key].data.numpy()

                        yield weight, bias, 'lstm_'+name

            # FC Layer
            if isinstance(layer, nn.Linear):
                name = layer._get_name()
                weight = layer.state_dict()['weight'].data.numpy()
                bias = layer.state_dict()['bias'].data.numpy()

                yield weight, bias, name

    def visualize_weights(self):
        # todo: delete 3,4 and make this method generic

        wbn = self.weight_bias_name()
        weights, biases, names = list(zip(*[(weight, bias, name) for weight, bias, name in wbn]))

        f, axarr = plt.subplots(3, 4)  # (nrows, ncols)

        def _process_weight(ax, name, weight):
            ax.set_title(name)
            ax.imshow(weight, )
            # ax.colorbar()

        def _maybe_reshape(arr):
            if np.ndim(arr) == 1:
                return np.expand_dims(arr, axis=1)
            if arr.shape[0] == 1:
                return np.transpose(arr)
            return arr

        idx = 0
        for (name, weight, bias) in zip(names, weights, biases):
            _process_weight(axarr[idx // 4][idx % 4], name + '_weight', _maybe_reshape(weight))
            idx += 1
            _process_weight(axarr[idx // 4][idx % 4], name + '_bias', _maybe_reshape(bias))
            idx += 1

        return f

class CNN(GenericModel):
    def __init__(self, config):

        # todo: add params to CNN
        GenericModel.__init__(self)
        self.device = config.DEVICE


        self.conv1 = nn.Conv2d(in_channels=config.INPUT_SIZE, out_channels=10,
                               kernel_size=5, stride=1,
                               padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20,
                               kernel_size=5, stride=1,
                               padding=0, dilation=1, groups=1, bias=True)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, config.OUTPUT_SIZE)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), 0.005)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def dummy_input(self):
        return Variable(torch.rand(100, 1, 28, 28)).to('cuda')


def class_by_name(name):
    if name == "CNN":
        return CNN

    if name == "LSTM":
        return LSTM



class LSTM(GenericModel):
    """
    """

    def __init__(self, input_size, seq_length, num_layers, out_size, hidden_size, batch_size, device):
        GenericModel.__init__(self)

        self.input_size = input_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.out_size = out_size
        self.batch_size = batch_size

        self.hidden_size = hidden_size

        self.name = 'LSTM'
        self.device = device

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
                            bias=True,
                            batch_first=False,
                            dropout=0.5,
                            bidirectional=False)

        self.fc = nn.Sequential(nn.Linear(in_features=self.hidden_size, out_features=100),
                                nn.BatchNorm1d(num_features=100),  # todo: test batchnorm
                                nn.SELU(),  # todo: test RELU vs SELU
                                nn.Linear(in_features=100, out_features=100),
                                nn.BatchNorm1d(num_features=100),
                                nn.SELU(),
                                nn.Linear(in_features=100, out_features=10),
                                nn.BatchNorm1d(num_features=10),
                                nn.SELU(),
                                nn.Linear(in_features=10, out_features=self.out_size))

        # self.softmax = nn.Softmax(dim=1)

        self.hidden = None

        # todo: I have found that initialization destroys the learning process. Think why and fixit.
        # self.initialize()

    def initialize(self):
        for w in self.lstm.parameters():
            init.normal_(w)  # inplace

        for w in self.fc.parameters():
            init.normal_(w)  # inplace

    def init_hidden(self, batch_size=None):
        """
        Returns:
            (Variable,Variable): (h_0, c_0)
        """
        if batch_size is None:
            batch_size = self.batch_size
        (h0, c0) = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)),  # h_0
                    Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)))  # c_0

        return h0.to(self.device), c0.to(self.device)

    def detach(self):
        # detach to not backpropagate whole lstm network
        # todo: actually below lines should be like self.model.hidden[0].detach_() aka inplace version. but not it working okey..
        self.hidden[0].detach()
        self.hidden[1].detach()

    def forward(self, x):

        # Reshape input
        # x shape: (seq_len, batch, input_size)
        # hidden shape:(num_layers * num_directions,batch_size, hidden_size)

        # reset the LSTM hidden state. Must be applied before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence

        # send batch size to auto update hiddens
        batch_size = x.shape[0]
        self.hidden = self.init_hidden(batch_size=batch_size)

        x = x.view(self.seq_length, -1, self.input_size)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # return last sequence
        # output of shape (seq_len, batch, num_directions * hidden_size)
        out = lstm_out[-1]
        fc_out = self.fc(out)

        # soft_out = self.softmax(fc_out)
        return fc_out

    def dummy_input(self):
        return Variable(torch.rand(self.batch_size, 1, self.input_size, self.seq_length)).type(torch.FloatTensor).to(self.device)







if __name__ == "__main__":
    pass

