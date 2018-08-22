import torch
from torch import nn
from torch.autograd import Variable



class LSTM(nn.Module):
    """
    """

    def __init__(self, input_size, seq_length, num_layers, out_size,hidden_size, batch_size, use_cuda):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.out_size = out_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size

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
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=True,
                            batch_first=False,
                            dropout=0.2,
                            bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(in_features=10, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=self.out_size)
        )

        self.softmax = nn.Softmax(dim=1)

        self.hidden = None



    def init_hidden(self, batch_size=None):
        """
        Returns:
            (Variable,Variable): (h_0, c_0)
        """
        if batch_size is None:
            batch_size = self.batch_size
        (h0,c0) = (Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)),  # h_0
                Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)))  # c_0

        if self.use_cuda:
            return h0.cuda(), c0.cuda()

        return h0, c0

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








from torch.nn.modules.module import _addindent
import torch
import numpy as np
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr



if __name__ == "__main__":
    model = LSTM(input_size=4,
                 seq_length=256,
                 num_layers=1,
                 out_size=1,
                 batch_size=2,
                 use_cuda=True).cpu()
    # Test
    print(torch_summarize(model))