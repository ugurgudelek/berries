# -*- coding: utf-8 -*-
# @Time   : 3/30/2020 8:18 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : chatter_experiment.py


from dataset.chatter_image import ChatterImage
from model.cnn import CNN


from trainer.classifier import ClassifierTrainer
import torch
import numpy as np
import os

if __name__ == "__main__":
    SEED = 47
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.multiprocessing.freeze_support()

    hyperparams = dict(lr=0.001, train_batch_size=2, test_batch_size=2, epoch=10)
    params = dict(log_interval=10, result_path='../results/chatter_cnn_images')

    dataset = ChatterImage(root='D:/YandexDisk/machining/chatter_cnn_images/Chatter_cnn',
                           # transform=transforms.Compose([transforms.ToTensor(),
                           #                               transforms.Normalize((0.1307,), (0.3081,))]),
                           )
    model = CNN(in_channels=1, out_channels=2)

    trainer = ClassifierTrainer(model=model, dataset=dataset, hyperparams=hyperparams, params=params)

    trainer.fit()


    trainer.history.save(fpath=params['result_path'])
    trainer.history.plot(fpath=params['result_path'])

# trainer.predict(data=dataset.testset.data)
#
# trainer.score(data=dataset.testset.data, targets=dataset.testset.targets)
