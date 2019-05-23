import torch
from numpy.core.multiarray import ndarray
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch import optim
import math

from abc import ABCMeta, abstractmethod


class GenericModel(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        nn.Module.__init__(self)

        self.train_loss = 0.
        self.validation_loss = 0.

    # ========== model save methods ==========
    def _to_onnx(self, directory):
        # todo: below line not working right now. so check:
        # https://github.com/lanpa/tensorboardX/issues/166
        # estimator.writer.add_graph(model, (dummy_input,))

        assert 'dummy_input' in dir(self), 'dummy_input method should be implemented in model class!'
        torch.onnx.export(self, self.dummy_input(), os.path.join(directory, 'model.onnx'), verbose=True)

    def _to_txt(self, directory):
        with open(os.path.join(directory, 'model.txt'), 'w') as f:
            f.write(self.__str__())

    def save(self, directory):
        self._to_onnx(directory=directory)
        self._to_txt(directory=directory)

    def save_checkpoint(self, directory, epoch):
        # save model
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   os.path.join(directory, 'checkpoints', str(epoch), "model-optim.pth"))

    def load_checkpoint(self, directory, epoch):
        # load model

        map_location = f"{self.device.type}:{self.device.index}"
        if self.device.type == 'cpu':
            map_location = self.device.type

        checkpoint = torch.load(os.path.join(directory, 'checkpoints', str(epoch), "model-optim.pth"),
                                map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.device.type == 'cuda':
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    # ========= model utility methods ========
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
                num_lstm_layer = layer.state_dict().keys().__len__() // 4  # input,hidden,weight,bias for each

                for i_h in ['i', 'h']:
                    for layer_num in range(num_lstm_layer):
                        name = '{i_or_h}h_l{layer_num}'.format(i_or_h=i_h, layer_num=layer_num)
                        weight_key = 'weight_' + name
                        bias_key = 'bias_' + name

                        weight = layer.state_dict()[weight_key].data.numpy()
                        bias = layer.state_dict()[bias_key].data.numpy()

                        yield weight, bias, 'lstm_' + name

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

    # ========= model fit,validate,predict methods ========
    def _pass(self, X, y, train):
        """
        Fit the model to data matrix X and target(s) y.
        If this call for validation or test, change model mode to evaluation
        This is necessary because dropout and batch normalization should behave
        differently on evaluation mode

        Args:
            X (torch.Tensor): training_data
            y (torch.Tensor): training_label
            train (bool): True = training_mode, False = validation_mode

        Returns:
            loss (torch.Tensor): calculated loss comes from self.criterion

        """

        # forward
        outputs = self.forward(X)
        # loss
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):  # classification problem
            y = y.long()
        loss = self.criterion(outputs, y)

        # if loss._grad_fn is not None:  # means we are in training mode
        # oor we can one more argument to hold train or validation mode.
        if train:
            # do not let pytorch accumulates gradients.
            self.optimizer.zero_grad()

            # backward -> calculate gradient over w.
            loss.backward(retain_graph=False)
            # optimize -> w = w - alpha*gradient
            self.optimizer.step()

        # todo: i dont know why we need to detach?
        # detach first hiddens of previous iteration
        # if isinstance(self, LSTM):
        #     self.detach()

        return loss.item()

    def fit(self, X, y):
        self.train_loss = self._pass(X, y, train=True)

    def validate(self, X, y):
        self.train(False)
        with torch.no_grad():
            self.validation_loss = self._pass(X, y, train=False)
        self.train(True)

    # todo: ======== need to check these methods ========
    def predict(self, X, classification=False, return_sequences=False):
        if return_sequences and classification:
            raise Exception("When return_sequences is True, classification output cannot be processed")
        if X.dim() != 3:
            raise Exception(f"Dim of X should be 3 (batch,seq,input) but {X.dim()}")
        self.train(False)
        with torch.no_grad():
            outputs = self.forward(X, return_sequences).cpu()
        self.train(True)
        if classification:
            _, outputs = torch.max(outputs, 1)
        return outputs.data.numpy()

    def predict_one_step_ahead(self, X_batch, classification=False, return_sequences=False):
        """Can only work with (batch,seq,input):(1,:,:).
        And predicts one step ahead.
        Returns:
            predictions_batch (np.array(X_batch.shape[0], self.hidden_size))
            """
        predictions_batch = np.zeros((X_batch.shape[0], self.hidden_size))
        for step, x in enumerate(X_batch):
            x = torch.Tensor([x]).to(self.device).float()


            predictions = self.predict(x,
                                       classification=classification,
                                       return_sequences=return_sequences)
            predictions_batch[step, :] = predictions
        return predictions_batch

    def predict_future(self, initial_x, n_steps, classification=False):
        """
        Only works with stateful system.
        Args:
            initial_x (tuple(-1,-1,1)): Start future prediction from here.
        """
        seq_len = initial_x.shape[1]

        # should be flatten to predict one by one
        initial_x = initial_x.reshape((-1, 1, 1))

        future_predictions = np.zeros((n_steps*seq_len, self.hidden_size))


        # feed given sequence, until all consumed
        for step in range(initial_x.shape[0]):
            input = initial_x[step, :, :].reshape((1, 1, 1))
            output = self.predict_one_step_ahead(input, classification=classification,
                                       return_sequences=False)
            future_predictions[step, :] = output

        # then start to predict future
        input = future_predictions[initial_x.shape[0]-1, 0].reshape((1, 1, 1)) # start from last output
        for step in range(initial_x.shape[0], n_steps):
            output = self.predict_one_step_ahead(input, classification=classification,
                                                 return_sequences=False)
            future_predictions[step, :] = output
            input = output[:, 0].reshape((1, 1, 1))
        return future_predictions


    def predict_batch(self, X, stateful=False):
        with torch.no_grad():
            self.train(False)
            if not stateful:
                self.init_hidden(batch_size=X.shape[0], inplace=True)
            outputs = self.forward(X)
            self.train(True)
        return outputs.data.numpy()



    def score(self, X, y):
        # score(X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels.

        predicted = self.predict(X)
        correct = (predicted == y).sum().item()
        return correct / (y.size()[0])

    def predict_log_proba(self, X):
        # predict_log_proba(X)	Return the log of probability estimates.
        pass

    def predict_proba(self, X):
        # predict_proba(X)	Probability estimates.
        pass

    def set_params(self, **params):
        # set_params(**params)	Set the parameters of this estimator.
        pass

    def get_params(self, deep=None):
        # get_params([deep])	Get parameters for this estimator.
        pass


# # ======== fit methods (never call these directly!) ========
# def _on_dataloader(self, dataloader, train):
#     """Never call this function directly!"""
#
#     losses = np.array([])
#     for step, (xs, ys) in enumerate(dataloader):
#         xs = xs.unsqueeze(dim=1)
#         ys = ys.unsqueeze(dim=1)
#         xs = Variable(xs.float(), requires_grad=False)
#         ys = Variable(ys.float(), requires_grad=False)
#
#         if train:
#             output,loss = self.train_on_batch(xs, ys)
#         else:
#             output,loss = self.validate_on_batch(xs, ys)
#
#         losses = np.append(losses, loss.item())
#
#     if train:
#         self.train_loss = losses.mean()
#     if not train:
#         self.valid_loss = losses.mean()
#
# def _on_batch(self, xs, ys, train):
#     """Never call this function directly!"""
#     # if this call for validation or test, change model mode to evaluation
#     # this is necessary because dropout and batch normalization should behave differently on evaluation mode
#     if not train:
#         self.eval()
#
#     self.optimizer.zero_grad()  # pytorch accumulates gradients.
#
#     xs = xs.to(self.device)
#     # forward
#     output = self.forward(xs)
#     loss = None
#
#     if ys is not None:
#         ys = ys.to(self.device)
#
#         # loss
#         if isinstance(self.criterion, nn.NLLLoss):
#             ys = ys.long()
#         loss = self.criterion(output, ys.squeeze(dim=1).long())
#
#         if train:
#
#             # backward
#             loss.backward(retain_graph=True)
#             #optimize
#             self.optimizer.step()
#
#     # detach first hiddens of previous iteration
#     if isinstance(self, LSTM):
#         self.detach()
#
#     # if this call for validation or test, model mode has changed to eval before,
#     # now we need to revert this behaviour
#     if not train:
#         self.train()
#
#     return output, loss

#  ======== Sklearn type model methods ========


class LSTM(GenericModel):
    """
    """

    def __init__(self, criterion, optimizer, input_size, num_layers, out_size, hidden_size, batch_size,
                 device, stateful=True, hidden_reset_period=None):
        GenericModel.__init__(self)

        self.name = 'LSTM'
        self.input_size = input_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.batch_size = batch_size

        self.hidden_size = hidden_size

        self.device = device
        self.stateful = stateful
        self.step = 0
        if stateful and hidden_reset_period is None:
            raise Exception("When stateful is True, you should pass hidden_reset_period")
        self.hidden_reset_period = hidden_reset_period

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
                            batch_first=True,
                            dropout=0.,  # default:0 means no probability
                            bidirectional=False)

        self.fc = nn.Sequential(nn.Linear(in_features=self.hidden_size, out_features=self.out_size))


        self.hidden = self._init_hidden()

        self.init_weights()

        self.criterion = criterion

        self.optimizer = None
        if optimizer['name'] == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), optimizer['lr'])
        if optimizer['name'] == 'Adadelta':
            self.optimizer = optim.Adadelta(self.parameters())
        if optimizer['name'] == 'SGD':
            self.optimizer = optim.SGD(self.parameters(),
                                       lr=optimizer['lr'],
                                       momentum=optimizer['momentum'],
                                       nesterov=optimizer['nesterov'])
        if self.optimizer is None:
            raise Exception('optimizer should be defined.')

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


    def _init_hidden(self, batch_size=None):
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
        self.hidden = self._init_hidden(batch_size=batch_size)

    def forward(self, x, return_sequences=False, predict=False):
        """

        Args:
            x (torch.Tensor): shape:(batch, seq, input)

        Returns:

        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        input_size = x.shape[2]

        # required x shape: (batch_size, seq_len, input_size) because batch_first=True
        # required hidden shape:(num_layers * num_directions, batch_size, hidden_size)

        # hidden reset -> if not applied: the LSTM will treat a new batch as a continuation of a sequence (Stateful LSTM)
        #              -> if applied: Stateless LSTM
        # See. [Stateful vs Stateless LSTM](http://philipperemy.github.io/keras-stateful-lstm/)

        # if self.stateful:
        #     if (self.hidden_reset_period != -1) and ((self.step % self.hidden_reset_period) == 0):
        #         self.reset_states()
        #     current_hidden = self.previous_hidden
        #     self.step += seq_len  # increment step to determine to reset hidden
        # else:
        #     current_hidden = self._init_hidden(batch_size=batch_size)

        if not self.stateful:
            self.hidden = self._init_hidden(batch_size=batch_size)
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

    # def dummy_input(self):
    #     return Variable(torch.rand(self.batch_size, self.input_size, self.seq_length)).type(torch.FloatTensor).to(self.device)


class LinearRegression():
    pass


class ARIMA():
    pass


class GARCH():
    pass
