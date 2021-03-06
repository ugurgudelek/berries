# -*- coding: utf-8 -*-
# @Time   : 5/28/2020 3:56 AM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : metrics.py

import torch
from torch import nn


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        loss = self.mse(yhat.detach(), y)
        return loss.cpu().numpy().item()


class RMSE(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat.detach(), y) + self.eps)
        return loss.cpu().numpy().item()


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, yhat, y):
        loss = self.mae(yhat.detach(), y)
        return loss.cpu().numpy().item()


class MAPE(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mae = nn.L1Loss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = 100*self.mae(yhat.detach()/(y+self.eps), y/(y+self.eps))
        return loss.cpu().numpy().item()


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        return 100*torch.mean((torch.argmax(yhat.detach(), dim=1) == y).float()).cpu().numpy().item()


class R2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        dividend = ((yhat.detach() - yhat.detach().mean())*(y-y.mean())).sum()
        divisor = (y.size(0) - 1) * yhat.detach().std() * y.std()
        corr = torch.div(dividend, divisor)
        r2 = torch.pow(corr, 2)
        return r2.cpu().numpy().item()


if __name__ == "__main__":
    print(0.9729, R2().forward(yhat=torch.FloatTensor([2, 8, 10, 13, 18, 20]),
                               y=torch.FloatTensor([3, 8, 10, 17, 24, 27])))