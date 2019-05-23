import torch
import pandas as pd
import datetime

import datasets
import models
from experiment import Experiment

VERBOSE = 1
EPOCH_SIZE = 1000
TRAIN_TEST_RATIO = 0.8
RANDOM_SEED = 7


def get_device():
    device = torch.device('cpu')
    # if torch.cuda.is_available() and not (torch.cuda.get_device_name(0) == 'GeForce GT 650M'):
    #     device = torch.device('cuda:0')
    # else:
    #     print('USE_CUDA is set to False because this GPU is not available or too old to support.')
    return device

def get_datetime_stamp():
    d = datetime.datetime.now()
    return '-'.join(map(str, [d.day, d.month, d.year, d.hour, d.minute]))


# region Stateful Experiments

class StatefulExperiments:

    # region longmemory_debug_experiment
    @staticmethod
    def longmemory_debug_experiment():
        # Starting to converge around epoch=50 on regression, epoch=23 on classification
        dataset_name = 'LongMemoryDebugDataset'
        device = get_device()
        window_size = 20
        subseq_size = 1
        predict_n_step = 40

        # example data: 1: ([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] -> [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        #               2: ([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] -> [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        dataset = datasets.LongMemoryDebugDataset(window_size=window_size,
                                                  sample_size=20,
                                                  train_test_ratio=TRAIN_TEST_RATIO,
                                                  subseq_size=subseq_size)
        criterion = torch.nn.CrossEntropyLoss()

        hidden_reset_period = window_size // subseq_size
        if predict_n_step % window_size != 0:
            raise Exception("self.PREDICT_N % self.WINDOW_SIZE should be zero")
        model = models.LSTM(criterion=criterion,
                            optimizer={'name': 'Adam', 'lr': 0.001},
                            input_size=1,
                            out_size=2,
                            num_layers=1,
                            hidden_size=1,
                            batch_size=1,
                            device=device,
                            stateful=True,
                            hidden_reset_period=hidden_reset_period).to(device)

        experiment = Experiment.maybe_resume(experiment_dir=f'../experiment/{dataset_name}/{get_datetime_stamp()}',
                                             epoch=None,
                                             model=model,
                                             dataset_params={'dataset': dataset,
                                                             'dataset_name': dataset_name,
                                                             'train_batch_size': 1,
                                                             'valid_batch_size': 1},
                                             device=device,
                                             classification=True,
                                             sequence_sample=True,
                                             predict_n_step=predict_n_step,
                                             future_predict_initial_step_size=2,
                                             verbose=VERBOSE,
                                             seed=RANDOM_SEED)

        return experiment
    # endregion

    # region sinewave_experiment
    @staticmethod
    def sinewave_experiment():
        dataset_name = 'SinewaveDataset'
        device = get_device()
        subseq_size = 1
        predict_n_step = 60
        steps_per_cycle = 60
        number_of_cycles = 40
        random_factor = 0.

        dataset = datasets.StatefulTimeseriesDataset(
            dataset=datasets.GenericDataset.noisy_sin(steps_per_cycle=steps_per_cycle,
                                                      number_of_cycles=number_of_cycles,
                                                      random_factor=random_factor)['sin_t'].values,
            window_size=subseq_size,
            train_test_ratio=TRAIN_TEST_RATIO)

        criterion = torch.nn.MSELoss()

        # hidden_reset_period = -1  # infinite time series
        hidden_reset_period = steps_per_cycle  # reset on each new period
        if (predict_n_step*subseq_size) % steps_per_cycle != 0:
            raise Exception("predict_n_step % steps_per_cycle should be zero")
        model = models.LSTM(criterion=criterion,
                            optimizer={'name': 'SGD', 'lr': 0.001,'momentum':0.9, 'nesterov':True},
                            input_size=1,
                            out_size=1,
                            num_layers=1,
                            hidden_size=1,
                            batch_size=1,
                            device=device,
                            stateful=True,
                            hidden_reset_period=hidden_reset_period).to(device)

        experiment = Experiment.maybe_resume(experiment_dir=f'../experiment/{dataset_name}/{get_datetime_stamp()}',
                                             epoch=None,
                                             model=model,
                                             dataset_params={'dataset': dataset,
                                                             'dataset_name': dataset_name,
                                                             'train_batch_size': 1,
                                                             'valid_batch_size': 1},
                                             device=device,
                                             classification=False,
                                             sequence_sample=True,
                                             predict_n_step=predict_n_step,
                                             future_predict_initial_step_size=2,
                                             verbose=VERBOSE,
                                             seed=RANDOM_SEED)

        return experiment
    # endregion

    # region weather_experiment

    @staticmethod
    def weather_experiment():
        dataset_name = 'WeatherDataset'
        device = get_device()
        subseq_size = 1
        predict_n_step = 60
        hidden_reset_period = -1  # infinite time series

        data = pd.read_csv('../input/weather_2017.csv').loc[:, 'temperature'].values
        dataset = datasets.StatefulTimeseriesDataset(dataset=data,
                                                     window_size=subseq_size,
                                                     train_test_ratio=TRAIN_TEST_RATIO,
                                                     normalize=True)

        criterion = torch.nn.MSELoss()

        model = models.LSTM(criterion=criterion,
                            optimizer={'name': 'Adam', 'lr': 0.001},
                            input_size=1,
                            out_size=1,
                            num_layers=1,
                            hidden_size=20,
                            batch_size=1,
                            device=device,
                            stateful=True,
                            hidden_reset_period=hidden_reset_period).to(device)

        experiment = Experiment.maybe_resume(experiment_dir=f'../experiment/{dataset_name}/{get_datetime_stamp()}',
                                             epoch=None,
                                             model=model,
                                             dataset_params={'dataset': dataset,
                                                             'dataset_name': dataset_name,
                                                             'train_batch_size': 1,
                                                             'valid_batch_size': 1},
                                             device=device,
                                             classification=False,
                                             sequence_sample=True,
                                             predict_n_step=predict_n_step,
                                             future_predict_initial_step_size=2,
                                             verbose=VERBOSE,
                                             seed=RANDOM_SEED)

        return experiment
    # endregion

    # region energy_load_experiment
    @staticmethod
    def energy_load_experiment():
        dataset_name = 'EnergyLoadDataset'
        device = get_device()
        subseq_size = 1
        predict_n_step = 60
        hidden_reset_period = -1  # infinite time series

        data = pd.read_csv('../input/load.csv').loc[:, ['from', 'actual']].rename({'from': 'date'}, axis=1)
        data['date'] = data['date'].astype('datetime64[ns]')
        data = data.loc[data['date'].dt.year < 2011]  # only 2010
        data.index = data['date']
        data = data['actual']
        data = data.resample('12H').sum().values
        dataset = datasets.StatefulTimeseriesDataset(dataset=data,
                                                     window_size=subseq_size,
                                                     train_test_ratio=TRAIN_TEST_RATIO,
                                                     normalize=True)

        criterion = torch.nn.MSELoss()

        model = models.LSTM(criterion=criterion,
                            optimizer={'name': 'Adam', 'lr': 0.001},
                            input_size=1,
                            out_size=1,
                            num_layers=1,
                            hidden_size=20,
                            batch_size=1,
                            device=device,
                            stateful=True,
                            hidden_reset_period=hidden_reset_period).to(device)

        experiment = Experiment.maybe_resume(experiment_dir=f'../experiment/{dataset_name}/{get_datetime_stamp()}',
                                             epoch=None,
                                             model=model,
                                             dataset_params={'dataset': dataset,
                                                             'dataset_name': dataset_name,
                                                             'train_batch_size': 1,
                                                             'valid_batch_size': 1},
                                             device=device,
                                             classification=False,
                                             sequence_sample=True,
                                             predict_n_step=predict_n_step,
                                             future_predict_initial_step_size=2,
                                             verbose=VERBOSE,
                                             seed=RANDOM_SEED)

        return experiment
    # endregion

    # region simple_finance_experiment

    @staticmethod
    def simple_finance_experiment():
        dataset_name = 'SimpleFinanceDataset'
        device = get_device()
        subseq_size = 1
        predict_n_step = 60
        hidden_reset_period = -1  # infinite time series

        data = pd.read_csv('../input/spy.csv').iloc[:400, :]
        dataset = datasets.StatefulTimeseriesDataset(dataset=data['adjusted_close'].diff(periods=1).values[1:],
                                                     window_size=subseq_size,
                                                     train_test_ratio=TRAIN_TEST_RATIO,
                                                     squeeze=True)

        criterion = torch.nn.MSELoss()

        model = models.LSTM(criterion=criterion,
                            optimizer={'name': 'Adam', 'lr': 0.001},
                            input_size=1,
                            out_size=1,
                            num_layers=1,
                            hidden_size=20,
                            batch_size=1,
                            device=device,
                            stateful=True,
                            hidden_reset_period=hidden_reset_period).to(device)

        experiment = Experiment.maybe_resume(experiment_dir=f'../experiment/{dataset_name}/{get_datetime_stamp()}',
                                             epoch=None,
                                             model=model,
                                             dataset_params={'dataset': dataset,
                                                             'dataset_name': dataset_name,
                                                             'train_batch_size': 1,
                                                             'valid_batch_size': 1},
                                             device=device,
                                             classification=False,
                                             sequence_sample=True,
                                             predict_n_step=predict_n_step,
                                             future_predict_initial_step_size=2,
                                             verbose=VERBOSE,
                                             seed=RANDOM_SEED)

        return experiment

    # endregion

    # region finance_experiment

    @staticmethod
    def finance_experiment():
        dataset_name = 'FinanceDataset'
        device = get_device()
        predict_n_step = 60
        hidden_reset_period = -1  # infinite time series
        window_size = 10

        dataset = datasets.FinanceDataset(path='../input/spy.csv',
                                          train_test_ratio=TRAIN_TEST_RATIO,
                                          window_size=window_size,
                                          stride_size=window_size,
                                          look_ahead_size=5)

        criterion = torch.nn.MSELoss()

        model = models.LSTM(criterion=criterion,
                            optimizer={'name': 'Adam', 'lr': 0.001},
                            input_size=1,
                            out_size=1,
                            num_layers=5,
                            hidden_size=100,
                            batch_size=1,
                            seq_len=window_size,
                            device=device,
                            stateful=True,
                            hidden_reset_period=hidden_reset_period).to(device)

        experiment = Experiment.maybe_resume(experiment_dir=f'../experiment/{dataset_name}/{get_datetime_stamp()}',
                                             epoch=None,
                                             model=model,
                                             dataset_params={'dataset': dataset,
                                                             'dataset_name': dataset_name,
                                                             'train_batch_size': 1,
                                                             'valid_batch_size': 1},
                                             device=device,
                                             classification=False,
                                             sequence_sample=True,
                                             predict_n_step=predict_n_step,
                                             future_predict_initial_step_size=2,
                                             verbose=VERBOSE,
                                             seed=RANDOM_SEED)

        return experiment

    # endregion


# endregion


# region Stateless Experiments

class StatelessExperiments:

    TIMESERIES_TYPE = 'Stateless'
    # region longmemory_debug_experiment
    @staticmethod
    def longmemory_debug_experiment():
        # Starting to converge around epoch=50 on regression, epoch=100 on classification
        dataset_name = 'LongMemoryDebugDataset'
        device = get_device()
        window_size = 20
        subseq_size = 20
        predict_n_step = 40

        # example data: ([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        dataset = datasets.LongMemoryDebugDataset(window_size=window_size,
                                                  sample_size=200,
                                                  train_test_ratio=TRAIN_TEST_RATIO,
                                                  subseq_size=subseq_size)
        criterion = torch.nn.CrossEntropyLoss()

        hidden_reset_period = window_size // subseq_size
        if predict_n_step % window_size != 0:
            raise Exception("self.PREDICT_N % self.WINDOW_SIZE should be zero")
        model = models.LSTM(criterion=criterion,
                            optimizer={'name': 'Adam', 'lr': 0.01},
                            input_size=1,
                            out_size=2,
                            num_layers=1,
                            hidden_size=1,
                            batch_size=1,
                            seq_len=subseq_size,
                            device=device,
                            stateful=False,
                            hidden_reset_period=hidden_reset_period).to(device)

        experiment = Experiment.maybe_resume(experiment_dir=f'../experiment/{dataset_name}/{get_datetime_stamp()}',
                                             epoch=None,
                                             model=model,
                                             dataset_params={'dataset': dataset,
                                                             'dataset_name': dataset_name,
                                                             'train_batch_size': 1,
                                                             'valid_batch_size': 1},
                                             device=device,
                                             classification=True,
                                             sequence_sample=True,
                                             predict_n_step=predict_n_step,
                                             future_predict_initial_step_size=2,
                                             verbose=VERBOSE,
                                             seed=RANDOM_SEED)

        return experiment

    @staticmethod
    def sinewave_experiment():
        pass

    @staticmethod
    def weather_experiment():
        pass

    @staticmethod
    def load_experiment():
        pass

    @staticmethod
    def finance_experiment():
        dataset_name = 'StatelessFinanceDataset'
        device = get_device()
        predict_n_step = 60
        window_size = 10

        dataset = datasets.FinanceDataset(path='../input/spy.csv',
                                          train_test_ratio=TRAIN_TEST_RATIO,
                                          window_size=window_size,
                                          stride_size=window_size,
                                          look_ahead_size=5,
                                          classification=True)

        criterion = torch.nn.MSELoss()

        model = models.LSTM(criterion=criterion,
                            optimizer={'name': 'Adam', 'lr': 0.001},
                            input_size=1,
                            out_size=3,
                            num_layers=5,
                            hidden_size=100,
                            batch_size=1,
                            seq_len=window_size,
                            device=device,
                            stateful=False).to(device)

        experiment = Experiment.maybe_resume(experiment_dir=f'../experiment/{StatelessExperiments.TIMESERIES_TYPE}/{dataset_name}/{get_datetime_stamp()}',
                                             epoch=None,
                                             model=model,
                                             dataset_params={'dataset': dataset,
                                                             'dataset_name': dataset_name,
                                                             'train_batch_size': 1,
                                                             'valid_batch_size': 1,
                                                             'sampler': datasets.ImbalancedDatasetSampler(dataset.train_dataset)},
                                             device=device,
                                             classification=True,
                                             sequence_sample=True,
                                             predict_n_step=predict_n_step,
                                             future_predict_initial_step_size=2,
                                             verbose=VERBOSE,
                                             seed=RANDOM_SEED)

        return experiment


# endregion


if __name__ == "__main__":
    # StatefulExperiments.longmemory_debug_experiment().run(epoch_size=EPOCH_SIZE)
    StatefulExperiments.sinewave_experiment().run(epoch_size=EPOCH_SIZE)
    # StatefulExperiments.weather_experiment().run(epoch_size=EPOCH_SIZE)
    # StatefulExperiments.energy_load_experiment().run(epoch_size=EPOCH_SIZE)
    # StatefulExperiments.simple_finance_experiment().run(epoch_size=EPOCH_SIZE)
    # StatefulExperiments.finance_experiment().run(epoch_size=EPOCH_SIZE)

    # StatelessExperiments.longmemory_debug_experiment().run(epoch_size=EPOCH_SIZE)
    # StatelessExperiments.finance_experiment().run(epoch_size=EPOCH_SIZE)


