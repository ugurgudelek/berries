__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from berries.trainer.base import BaseTrainer
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


class LSTMTrainer(BaseTrainer):

    def __init__(self,
                 model,
                 metrics,
                 hyperparams,
                 params,
                 optimizer=None,
                 criterion=None,
                 logger=None):
        super().__init__(model, metrics, hyperparams, params, optimizer,
                         criterion, logger)


class CNNTrainer(BaseTrainer):

    def __init__(self,
                 model,
                 metrics,
                 hyperparams,
                 params,
                 optimizer=None,
                 criterion=None,
                 logger=None):
        super().__init__(model, metrics, hyperparams, params, optimizer,
                         criterion, logger)


class AETrainer(BaseTrainer):

    def __init__(self,
                 model,
                 metrics,
                 hyperparams,
                 params,
                 optimizer=None,
                 criterion=None,
                 logger=None):
        super().__init__(model, metrics, hyperparams, params, optimizer,
                         criterion, logger)

    def _encode(self, dataset):
        loader = self._to_loader(dataset, training=False)

        latents = []
        self._set_grad_enabled(False)
        for batch_ix, batch in enumerate(loader):
            data, target = self.handle_batch(batch)
            latent = self.model.latent(data)
            latents.append(latent)
        latents = torch.cat(latents, axis=0)
        return latents

    def encode(self, dataset):
        return self._encode(dataset).cpu().detach().numpy()


class VAETrainer(AETrainer):

    def __init__(self,
                 model,
                 metrics,
                 hyperparams,
                 params,
                 optimizer=None,
                 criterion=None,
                 logger=None):
        super().__init__(model, metrics, hyperparams, params, optimizer,
                         criterion, logger)

    def compute_loss(self, output, targets):
        targets = targets.permute(1, 0, 2)
        output = output.permute(1, 0, 2)
        joint_loss, recon_loss, kl_loss = self._rec(x_decoded=output,
                                                    x=targets,
                                                    loss_fn=self.criterion)
        return joint_loss

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the specified loss function

        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.model.lmbd.latent_mean, self.model.lmbd.latent_logvar

        kl_loss = -0.5 * \
            torch.mean(1 + latent_logvar -
                        latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss


class PredictorFromCompressorTrainer(BaseTrainer):

    def __init__(self,
                 model,
                 compressor,
                 metrics,
                 hyperparams,
                 params,
                 optimizer=None,
                 criterion=None,
                 logger=None):
        super().__init__(model, metrics, hyperparams, params, optimizer,
                         criterion, logger)

        self.compressor = compressor

    def handle_batch(self, batch):
        data = batch['data']
        target = batch['target'].squeeze()

        # cast data to a device
        data, target = data.to(self.device), target.to(self.device)

        # compress the information into
        latent = self.compressor.latent(data).detach()
        return latent, target