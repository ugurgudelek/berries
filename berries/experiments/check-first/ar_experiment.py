import torch
from dataset.timeseries import *
import matplotlib.pyplot as plt
from model.lstm import LSTM

from trainer.regressor import RegressorTrainer

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    hyperparams = {'lr': 1e-3,
                   'train_batch_size': 80,
                   'test_batch_size': 80,
                   'epoch': 500}
    params = dict(log_interval=100)

    dataset = ARDataset(num_datapoints=100, num_prev=10, test_size=0.2, noise_var=0)
    model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
    trainer = RegressorTrainer(model=model, dataset=dataset, hyperparams=hyperparams, params=params)

    trainer.fit()

    data = dataset.trainset.data
    targets = dataset.trainset.targets.numpy()
    pred = trainer.predict(data=data)
    plt.plot(targets, label='target')
    plt.plot(pred, label='pred')
    plt.legend()
    plt.show()
