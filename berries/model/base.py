from sklearn.base import BaseEstimator as SklearnBaseEstimator
from torch import nn
import torch


class BaseEstimator(SklearnBaseEstimator):
    # http://msmbuilder.org/development/apipatterns.html

    def summarize(self):
        return 'NotImplemented'


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def load_model_from_path(self, path, device):
        # load model
        map_location = f"{device.type}:{device.index}"
        if device.type == 'cpu':
            map_location = device.type

        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])

        return self
