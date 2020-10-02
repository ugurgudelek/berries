from sklearn.base import BaseEstimator as SklearnBaseEstimator
from torch import nn


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
