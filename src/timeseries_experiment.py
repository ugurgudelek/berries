import torch
import torch.nn as nn
from dataset.timeseries import *
import matplotlib.pyplot as plt
from model.lstm import LSTM
from torch.optim import Adam

from trainer.regressor import RegressorTrainer


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    hyperparams = {'lr':1e-3,
                   'train_batch_size': 80,
                   'test_batch_size':80,
                   'epoch':5000}
    params = dict(log_interval=10)

    dataset = ARDataset(num_datapoints=100, num_prev=20, test_size=0.2, noise_var=0)
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
#####################
# Generate data
#####################



#
#
#
#
# #####################
# # Train model
# #####################
#
# hist = np.zeros(hyperparams['epoch'])
#
# for t in range(hyperparams['epoch']):
#     # Initialise hidden state
#     # Don't do this if you want your LSTM to be stateful
#     model.hidden = None
#
#     # Forward pass
#     y_pred = model(X_train)
#
#     loss = loss_fn(y_pred, y_train)
#     if t % 100 == 0:
#         print("Epoch ", t, "MSE: ", loss.item())
#     hist[t] = loss.item()
#
#     # Zero out gradient, else they will accumulate between epochs
#     optimiser.zero_grad()
#
#     # Backward pass
#     loss.backward()
#
#     # Update parameters
#     optimiser.step()
#
# #####################
# # Plot preds and performance
# #####################
#
# plt.plot(y_pred.detach().numpy(), label="Preds")
# plt.plot(y_train.detach().numpy(), label="Data")
# plt.legend()
# plt.show()
#
# plt.plot(hist, label="Training loss")
# plt.legend()
# plt.show()