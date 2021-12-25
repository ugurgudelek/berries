# -*- coding: utf-8 -*-
# @Time   : 5/28/2020 3:56 AM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : metrics.py

import torch
from torch import nn
from torch.nn import functional as F


from torchmetrics import Metric


class MSE(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("nobs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        sum_squared_error = ((preds - target) ** 2).sum()
        nobs = target.numel()  # number of observations

        self.sum_squared_error += sum_squared_error
        self.nobs += nobs

    def compute(self):
        return self.sum_squared_error / self.nobs


class RMSE(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss.item()


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, yhat, y):
        loss = self.mae(yhat, y)
        return loss.item()


class MAPE(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mae = nn.L1Loss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = 100 * self.mae(yhat / (y + self.eps), y / (y + self.eps))
        return loss.item()


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        return 100 * torch.mean((torch.argmax(yhat, dim=1) == y).float()).item()


class AccuracyBar(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        return (100 * (1 - torch.mean(torch.abs(yhat - y)) / torch.mean(y))).item()


class R2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        dividend = ((yhat - yhat.mean()) * (y - y.mean())).sum()
        divisor = (y.size(0) - 1) * yhat.std() * y.std()
        corr = torch.div(dividend, divisor)
        r2 = torch.pow(corr, 2)
        return r2.item()


if __name__ == "__main__":
    print(
        0.9729,
        R2().forward(yhat=torch.FloatTensor([2, 8, 10, 13, 18, 20]), y=torch.FloatTensor([3, 8, 10, 17, 24, 27])),
    )
