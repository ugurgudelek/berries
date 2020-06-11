__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"


from skorch import NeuralNetClassifier
from main import CNN1D,CNN2D,NN, VibrationDataset, DatasetNumpyWrapper, img_transform
from sampler import ImbalancedDatasetSampler
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import LeaveOneOut
from torchvision.datasets import MNIST


# dataset = MNIST('../data/MNIST', download=True, transform=img_transform)
# X = dataset.train_data
# y= dataset.train_labels


dataset = VibrationDataset()
X = torch.from_numpy(np.array([_x.numpy() for _x, _ in dataset.dataset])).float()
y = torch.from_numpy((np.array([_y for _, _y in dataset.dataset]))).long()

print(f"Dataset label statistic:{y.numpy().mean()}")

loo = LeaveOneOut()

test_accs = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = NeuralNetClassifier(CNN1D,
                                max_epochs=500,
                                lr=1e-4,
                                device='cuda',
                                criterion=nn.CrossEntropyLoss,
                                batch_size=20,
                                train_split=None,
                                iterator_train__sampler=ImbalancedDatasetSampler(
                                    DatasetNumpyWrapper(X_train, y_train)
                                ),verbose=False)

    model.fit(X=X_train, y=y_train)

    pred_y = model.predict(X_test)
    test_accs.append(pred_y.item() == y_test.item())

    print(f"Test Acc:{np.array(test_accs).mean()}")



