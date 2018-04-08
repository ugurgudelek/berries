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

class Experiment:
    """

    """
    def __init__(self):

        config = Config()
        dataset = IndicatorDataset(stocks_dir=config.stocks_dir, stock_names=config.stock_names, label_after=config.label_after)
        estimator = Estimator(dataset=dataset,
                              model_config={'input_size': config.input_size,
                                            'seq_length': config.seq_length,
                                            'num_layers': config.num_layers,
                                            'out_size': config.out_size},
                              dataloader_config={'train_batch_size': config.train_batch_size,
                                                 'train_shuffle': config.train_shuffle,
                                                 'valid_batch_size': config.valid_batch_size,
                                                 'valid_shuffle': config.valid_shuffle})

        # fixme below
        experiment_dir = 'experiment/1'

        history = History(config.epoch_size, config.storage_names)
        checkpointer = Checkpointer(experiment_dir=experiment_dir)
        visualizer = Visualizer()







experiment = Experiment()