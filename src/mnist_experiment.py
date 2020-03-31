__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from dataset.mnist import MNIST
from model.cnn import CNN
from torchvision import transforms

from trainer.classifier import ClassifierTrainer
import torch


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    hyperparams = dict(lr=0.001, train_batch_size=1000, test_batch_size=1000, epoch=1)
    params = dict(log_interval=10)

    dataset = MNIST(root='../input/', transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))]))
    model = CNN(in_channels=1, out_channels=10)

    trainer = ClassifierTrainer(model=model, dataset=dataset, hyperparams=hyperparams, params=params)

    trainer.fit()

# trainer.predict(data=dataset.testset.data)
#
# trainer.score(data=dataset.testset.data, targets=dataset.testset.targets)
