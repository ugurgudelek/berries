__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import codecs
import numpy as np
import errno
from PIL import Image

from functools import reduce


class MNIST:
    """Placeholder class for MNISTInner
    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.


    """

    training_file = Path('training.pt')
    test_file = Path('test.pt')

    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
    ]

    classes = [
        '0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five',
        '6 - six', '7 - seven', '8 - eight', '9 - nine'
    ]

    def __init__(self, root, transform=None, target_transform=None):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists(self.processed_folder / self.training_file,
                                  self.processed_folder / self.training_file):
            print('Dataset not found. Trying to download ...')
            self.download()

        train_data, train_targets = torch.load(self.processed_folder /
                                               self.training_file)
        test_data, test_targets = torch.load(self.processed_folder /
                                             self.test_file)

        self.trainset = MNISTInner(data=train_data,
                                   targets=train_targets,
                                   transform=self.transform,
                                   target_transform=self.target_transform)
        self.testset = MNISTInner(data=test_data,
                                  targets=test_targets,
                                  transform=self.transform,
                                  target_transform=self.target_transform)

    @property
    def raw_folder(self):
        return self.root / self.__class__.__name__ / 'raw'

    @property
    def processed_folder(self):
        return self.root / self.__class__.__name__ / 'processed'

    def load_data(self, data_file):
        """

        Args:
            data_file (Path):

        Returns:

        """
        data, targets = torch.load(self.processed_folder / data_file)
        return data, targets

    def _check_exists(self, *args):
        return reduce(lambda a, b: os.path.exists(a) and os.path.exists(b),
                      args)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists(self.processed_folder / self.training_file,
                              self.processed_folder / self.test_file):
            return

        # download files
        try:
            os.makedirs(self.raw_folder)
            os.makedirs(self.processed_folder)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = self.raw_folder / filename
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(str(file_path).replace('.gz', ''),
                      'wb') as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (read_image_file(self.raw_folder /
                                        'train-images-idx3-ubyte'),
                        read_label_file(self.raw_folder /
                                        'train-labels-idx1-ubyte'))
        test_set = (read_image_file(self.raw_folder / 't10k-images-idx3-ubyte'),
                    read_label_file(self.raw_folder / 't10k-labels-idx1-ubyte'))
        with open(self.processed_folder / self.training_file, 'wb') as f:
            torch.save(training_set, f)
        with open(self.processed_folder / self.test_file, 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


class MNISTInner(Dataset):
    """
    Actual MNIST class to work with.
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
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            'data': img,
            'target': target,
        }

    def __len__(self):
        return len(self.data)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


if __name__ == "__main__":
    dataset = MNIST(root='../../input/', download=True)
    print()
