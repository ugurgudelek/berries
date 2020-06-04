# -*- coding: utf-8 -*-
# @Time   : 5/28/2020 3:56 AM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : __init__.py

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
        y_pred_tag = torch.round(torch.sigmoid(yhat.detach()))
        correct_results_sum = (y_pred_tag == y).sum().float()
        acc = correct_results_sum / y.shape[0]
        acc = torch.round(acc * 100)
        return acc.cpu().numpy().item()
