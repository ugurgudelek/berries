# -*- coding: utf-8 -*-
# @Time   : 3/30/2020 8:19 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : chatter_image.py


import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import codecs
import numpy as np
import errno
from PIL import Image

import pandas as pd
from tqdm import tqdm

from dataset.generic import GenericDataset


class ChatterImage(GenericDataset):
    """Placeholder class for MNISTInner
    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.


    """

    training_file = Path('training.pt')
    test_file = Path('test.pt')



    classes = ['0 - no_chatter', '1 - chatter']

    def __init__(self, root, transform=None, target_transform=None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.info_excel = self.read_info_xlsx()

        images, labels = self.load_data()
        self.images, self.labels = self.shuffle(images, labels)

        datasize = len(self.images)
        trainsize = int(datasize*0.75)
        testsize = datasize - trainsize

        train_data = self.images[:trainsize]
        train_targets = self.labels[:trainsize]

        test_data = self.images[trainsize:]
        test_targets = self.labels[trainsize:]



        self.trainset = ChatterImageInner(data=train_data, targets=train_targets,
                                   transform=self.transform,
                                   target_transform=self.target_transform)
        self.testset = ChatterImageInner(data=test_data, targets=test_targets,
                                  transform=self.transform,
                                  target_transform=self.target_transform)

        print("====== Trainset Info ======")
        print(self.trainset)

        print("====== Testset Info ======")
        print(self.testset)

    def shuffle(self, images, labels):
        p = np.random.permutation(len(images))
        return images[p], labels[p]

    def read_info_xlsx(self):
        excel = pd.read_excel(self.root/'Chatter_labels_CNN.xlsx', sheet_name='Tlusty_labels')
        excel = excel.loc[:, ['Name', 'Label', 'Used']]
        return excel

    def load_data(self):
        images = list()
        labels = list()

        with tqdm(total=self.info_excel['Used'].sum(), desc='Reading images...') as pbar:
            for ix, row in self.info_excel.iterrows():
                filename = row['Name']
                label = row['Label']
                use = row['Used']
                if use:
                    slotname = filename.split('_')[1]
                    if slotname == 'kanal1' or slotname == 'kanal2' or slotname == 'kanal3': #delete this line!
                        img_path = self.root/f'CNN-Inputs-Tlusty/{slotname}/{filename}'
                        img = Image.open(img_path).convert('L')
                        img = np.array(img)/255
                        images.append(img)
                        labels.append(label)
                        pbar.update(1)

        return np.array(images), np.array(labels)






class ChatterImageInner(Dataset):
    """
    Actual ChatterImage class to work with.
    Normalization should be implemented here!
    """

    def __init__(self, data, targets, transform, target_transform):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, ix):
        """
        Args:
            ix (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[ix], int(self.targets[ix])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.from_numpy(img).unsqueeze(dim=0).float(), target

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"""
                data.shape:{self.data.shape}
                targets.shape:{self.targets.shape}
                targets.mean: {self.targets.mean()}
        """




if __name__ == "__main__":
    dataset = ChatterImage(root='D:/YandexDisk/machining/chatter_cnn_images/Chatter_cnn')
    print()
