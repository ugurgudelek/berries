from config import ConfigLSTM
import datasets
import models
from experiment import Experiment
import numpy as np
import pandas as pd
import math

import torch
from torch import nn
import random

if __name__ == "__main__":
    """
    1. Implement Dataset Class
    2. Implement Model Class
    3. Configure configClass
    4. Pass config to experiment
    5. Run
    """

    # dataset = dataset.TimeSeriesARDataset(num_datapoints=10, test_size=0.2, num_prev=config.SEQ_LEN)
    # dataset = dataset.SequenceLearningManyToOne(seq_len=config.SEQ_LEN, seq_limit=300)
    # dataset = dataset.SequenceLearningManyToOneRotate(seq_len=config.SEQ_LEN, seq_limit=81, dataset_len=1000)
    # dataset = datasets.SineWaveDataset(seq_len=config.SEQ_LEN, train_test_ratio=config.TRAIN_TEST_RATIO)



    config = ConfigLSTM(dataset_name='LoadDataset')

    dataset = None
    if config.DATASET_NAME == 'SineWaveDataset':
        dataset = datasets.StatefulTimeseriesDataset(dataset=datasets.GenericDataset.noisy_sin(steps_per_cycle=config.STEPS_PER_CYCLE,
                                                                                               number_of_cycles=config.NUMBER_OF_CYCLES,
                                                                                               random_factor=config.RANDOM_FACTOR)['sin_t'].values,
                                                     window_size=config.SUBSEQ_SIZE,
                                                     train_test_ratio=config.TRAIN_TEST_RATIO)

    if config.DATASET_NAME == 'SPY':
        data = pd.read_csv('../input/spy.csv').iloc[:40, :]
        dataset = datasets.StatefulTimeseriesDataset(dataset=data['adjusted_close'].diff(periods=1).values[1:],
                                                     window_size=config.SUBSEQ_SIZE,
                                                     train_test_ratio=config.TRAIN_TEST_RATIO,
                                                     squeeze=True)

    if config.DATASET_NAME == 'LongMemoryDebugDataset':
        # example data: ([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        dataset = datasets.LongMemoryDebugDataset(window_size=config.WINDOW_SIZE,
                                                  sample_size=10,
                                                  train_test_ratio=config.TRAIN_TEST_RATIO,
                                                  subseq_size=config.SUBSEQ_SIZE)

    if config.DATASET_NAME == 'WeatherDataset':
        data = pd.read_csv('../input/weather_2017.csv').loc[:, 'temperature'].values
        dataset = datasets.StatefulTimeseriesDataset(dataset=data,
                                                     window_size=config.SUBSEQ_SIZE,
                                                     train_test_ratio=config.TRAIN_TEST_RATIO,
                                                     normalize=True)

    if config.DATASET_NAME == 'LoadDataset':
        data = pd.read_csv('../input/load.csv').loc[:, ['from', 'actual']].rename({'from':'date'}, axis=1)
        data['date'] = data['date'].astype('datetime64[ns]')
        data = data.loc[data['date'].dt.year < 2011]  # only 2010
        data.index = data['date']
        data = data['actual']
        data = data.resample('12H').sum().values
        dataset = datasets.StatefulTimeseriesDataset(dataset=data,
                                                     window_size=config.SUBSEQ_SIZE,
                                                     train_test_ratio=config.TRAIN_TEST_RATIO,
                                                     normalize=True)

    criterion = torch.nn.CrossEntropyLoss() if config.CLASSIFICATION else torch.nn.MSELoss()

    model = models.LSTM(criterion=criterion,
                        optimizer={'name': 'Adam', 'lr': config.LR},
                        input_size=config.INPUT_SIZE,
                        out_size=config.OUTPUT_SIZE,
                        num_layers=config.NUM_LAYERS,
                        hidden_size=config.HIDDEN_SIZE,
                        batch_size=config.TRAIN_BATCH_SIZE,
                        device=config.DEVICE,
                        stateful=config.STATEFUL,
                        hidden_reset_period=config.HIDDEN_RESET_PERIOD).to(config.DEVICE)
    print(config)
    print(model)
    print(dataset)

    experiment = Experiment.resume(directory='../experiment/LongMemoryDebugDataset/1551193121',
                                   config=config, model=model,
                                   dataset=dataset, sequence_sample=True, verbose=config.VERBOSE)

    # experiment.validate_before_run()




    experiment.run(epoch_size=config.EPOCH_SIZE)



