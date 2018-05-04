"""
Ugur Gudelek
run
ugurgudelek
08-Mar-18
finance-cnn
"""
from dataset import IndicatorDataset

from config import Config

import torch

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import torch.nn.functional as F

from history import History
from checkpoint import Checkpoint
from estimator import Estimator
from visualize import Visualizer

from loaddataset import LoadFullDataset

import io
import imageio

#todo: add logger
# todo: add reporter
# todo: plot and save graph

# import torch
# from torch.autograd import Variable
# from torch import nn
# from torch.utils.data import DataLoader
# from torch import optim
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import os
# import dill  # dill extends python’s pickle module for serializing and de-serializing python objects
# import shutil  # high level os functionality
#
# import gc
#
# from collections import defaultdict

# if we seed random func, they will generate same output everytime.



class Experiment:
    """

    """
    def __init__(self, config, dataset:LoadFullDataset, estimator, history, visualizer, epoch):
        self.config = config
        self.dataset = dataset
        self.estimator = estimator
        self.history = history
        self.visualizer = visualizer
        self.epoch = epoch

        if config.RANDOM_SEED is not None:
            torch.manual_seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)

    @classmethod
    def start_over(cls, config):

        dataset = LoadFullDataset(csv_path=config.INPUT_PATH,
                                  train_valid_ratio=config.TRAIN_VALID_RATIO,
                                  train_day=config.TRAIN_DAY,
                                  valid_day=config.VALID_DAY,
                                  seq_length=config.SEQ_LENGTH)

        estimator = Estimator(dataset=dataset,
                              model_config={'input_size': config.INPUT_SIZE,
                                            'seq_length': config.SEQ_LENGTH,
                                            'num_layers': config.NUM_LAYERS,
                                            'out_size': config.OUTPUT_SIZE,
                                            'batch_size': config.TRAIN_BATCH_SIZE},
                              dataloader_config={'train_batch_size': config.TRAIN_BATCH_SIZE,
                                                 'train_shuffle': config.TRAIN_SHUFFLE,
                                                 'valid_batch_size': config.VALID_BATCH_SIZE,
                                                 'valid_shuffle': config.VALID_SHUFFLE},
                              use_cuda=config.USE_CUDA)


        history = History(config.EPOCH_SIZE, config.STORAGE_NAMES)
        visualizer = Visualizer()

        epoch = 0

        return Experiment(config, dataset, estimator, history, visualizer, epoch)

    @classmethod
    def resume(self, experiment_path, config):
        ckpt_path = Checkpoint.get_latest_checkpoint(experiment_path)

        ckpt = Checkpoint.load(ckpt_path)


        experiment = Experiment.start_over(config)

        experiment.estimator.model = ckpt.model
        experiment.estimator.optimizer = ckpt.optimizer
        experiment.epoch = ckpt.epoch + 1
        experiment.history = ckpt.history

        # todo: improve with more proper way
        # experiment.visualizer.container['tloss'] = experiment.history.container[epoch]['train']['loss']
        # experiment.visualizer.container['vloss'] = experiment.history.container[epoch]['valid']['loss']

        return experiment

    def do(self):

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # plt.show(block=False)
        for self.epoch in range(self.epoch, self.config.EPOCH_SIZE):
            print('epoch : {}'.format(self.epoch))

            # Estimate - Train & Validate
            (toutputs, tloss, voutputs, vloss) = self.estimator.run_epoch(self.epoch)

            # Checkpoint
            Checkpoint(model=self.estimator.model, optimizer=self.estimator.optimizer,
                       epoch=self.epoch, history=self.history,
                       experiment_dir=self.config.EXPERIMENT_DIR).save()

            # Sample Predict
            ix, (pX, py) = self.estimator.dataset.train_dataset.get_sample()
            prediction = self.estimator.predict(pX)

            # Visualize
            im = self.visualizer.prediction_to_image(actual=py, prediction=prediction[0, :], im_title='TLoss:{:0.5f} || VLoss:{:0.5f} || Epoch:{} ||ix:{}'.format(tloss, vloss, self.epoch, ix))

            # Report to Tensorboard
            self.estimator.writer.add_image('prediction image', im, self.epoch)
            self.estimator.writer.add_scalar('training_loss', tloss, self.epoch)
            self.estimator.writer.add_scalar('validation_loss', vloss, self.epoch)

            self.estimator.writer.add_text('Text', 'text logged at step: {}'.format(self.epoch), self.epoch)

            self.estimator.writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), self.epoch)

            # self.visualizer.append_data('tloss', tloss)
            # self.visualizer.append_data('vloss', vloss)
            # self.visualizer.visualize(self.epoch)
            # self.visualizer.report()

            # todo: do we need this really? not sure.
            # Save
            # self.history.append(epoch=epoch, phase='train', name='loss', value=tloss)
            # self.history.append(epoch=epoch, phase='valid', name='loss', value=vloss)








config = Config()
experiment = Experiment.start_over(config)
# experiment = Experiment.resume(config.EXPERIMENT_DIR, config)
experiment.do()
sample = experiment.dataset.train_dataset.get_sample()
print()
# experiment.estimator.dataset.train_dataset.__getitem__()

# config = Config()
#
# estimator = LoadEstimator(config=config, resume=config.RESUME)
#
# if config.RESUME:
#     res_dict = estimator.test()
#     print()
# else:
#     estimator.train(epoch_size=config.EPOCH_SIZE)