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
from checkpoint import Checkpointer
from estimator import Estimator
from visualize import Visualizer

#todo: add logger
# todo: add reporter
# todo: plot and save graph



class Experiment:
    """

    """
    def __init__(self, config, dataset, estimator, experiment_dir, history, checkpointer, visualizer, epoch):
        self.config = config
        self.dataset = dataset
        self.estimator = estimator
        self.experiment_dir = experiment_dir
        self.history = history
        self.checkpointer = checkpointer
        self.visualizer = visualizer
        self.epoch = epoch

    @classmethod
    def start_over(cls):
        config = Config()
        dataset = IndicatorDataset(stocks_dir=config.stocks_dir, stock_names=config.stock_names,
                                   label_after=config.label_after)
        estimator = Estimator(dataset=dataset,
                              model_config={'input_size': config.input_size,
                                            'seq_length': config.seq_length,
                                            'num_layers': config.num_layers,
                                            'out_size': config.out_size},
                              dataloader_config={'train_batch_size': config.train_batch_size,
                                                 'train_shuffle': config.train_shuffle,
                                                 'valid_batch_size': config.valid_batch_size,
                                                 'valid_shuffle': config.valid_shuffle})

        # todo: fix below
        experiment_dir = '../experiment/2'

        history = History(config.epoch_size, config.storage_names)
        checkpointer = Checkpointer(experiment_dir=experiment_dir)
        visualizer = Visualizer()

        epoch = 0

        return Experiment(config, dataset, estimator, experiment_dir, history, checkpointer, visualizer, epoch)

    @classmethod
    def resume(self, experiment_path):
        ckpt_path = Checkpointer.get_latest_checkpoint(experiment_path)

        (model, optimizer, epoch, history) = Checkpointer.load(ckpt_path)

        experiment = Experiment.start_over()

        experiment.estimator.model = model
        experiment.estimator.optimizer = optimizer
        experiment.epoch = epoch
        experiment.history = history

        # todo: improve with more proper way
        experiment.visualizer.container['tloss'] = experiment.history.container[epoch]['train']['loss']
        experiment.visualizer.container['vloss'] = experiment.history.container[epoch]['valid']['loss']

        return experiment

    def do(self):
        for epoch in range(self.epoch, self.config.epoch_size):
            # Estimate - Train & Validate
            (toutputs, tloss, voutputs, vloss) = self.estimator.run_epoch(epoch)
            # Visualize
            self.visualizer.append_data('tloss', tloss)
            self.visualizer.append_data('vloss', vloss)
            self.visualizer.visualize()
            self.visualizer.report()
            # Save
            self.history.append(epoch=epoch, phase='train', name='loss', value=tloss)
            self.history.append(epoch=epoch, phase='valid', name='loss', value=vloss)
            # Checkpoint
            self.checkpointer.save(epoch=epoch, model=self.estimator.model,
                              optimizer=self.estimator.optimizer, history=self.history)







experiment = Experiment.start_over()
# experiment = Experiment.resume('../experiment/2')
experiment.do()