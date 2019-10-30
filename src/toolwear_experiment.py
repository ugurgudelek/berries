import torch
import torch.nn as nn
from dataset.toolwear import ToolWearDataset
import matplotlib.pyplot as plt
from model.lstm import LSTM
from torch.optim import Adam

from trainer.regressor import RegressorTrainer

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    hyperparams = {'lr': 1e-3,
                   'train_batch_size': 1000,
                   'test_batch_size': 1000,
                   'epoch': 50}
    params = dict(log_interval=1)

    dataset = ToolWearDataset()
    model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
    trainer = RegressorTrainer(model=model, dataset=dataset, hyperparams=hyperparams, params=params)

    trainer.fit()

    # d0,t0 = dataset.trainset.__getitem__(0)
    # d1,t1 = dataset.trainset.__getitem__(1)
    # data = torch.cat([d0.unsqueeze(1), d1.unsqueeze(1)], dim=1)
    # targets = torch.cat([t0.unsqueeze(1), t1.unsqueeze(1)], dim=1).numpy()
    # pred = trainer.predict(data=data)
    # plt.plot(targets[:, 0], label='target')
    # plt.plot(pred[:, 0], label='pred')
    # plt.legend()
    # plt.show()

    trainer.history.to_dataframe(phase='train').plot(x='epoch', y='loss')
    plt.show()
