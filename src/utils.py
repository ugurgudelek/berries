"""Utility methods"""
import numpy as np

class Bucket:
    """This is variation of queue data structure.
    It holds first few items. Whenever new item added, oldest will be deleted."""

    def __init__(self, size):
        self.size = size
        self.container = []

    def __len__(self):
        return len(self.container)

    def full(self):
        return self.__len__() == self.size

    def put(self, data):
        if self.full():  # throw older item if full
            self.container.pop(0)

        # append new as last item
        self.container.append(data)

    def peek(self, index=-1):
        return self.container[index]

    def get_all_bucket(self):
        return self.container

    def average(self):
        return sum(self.container) / self.size

    def max(self):
        return max(self.container)

    def min(self):
        return min(self.container)


    def flush(self):
        self.container = []

    def is_empty(self):
        return self.__len__() == 0

    def __str__(self):
        string = ""
        string += str(self.container)
        return string

    def __repr__(self):
        return self.__str__()

def normalize_column_based(image):
    """Requires 2D image"""
    image = np.array(image)
    for i in range(image.shape[1]):  # for each column
        image[:, i] = normalize(image[:, i])
    return image
def normalize(arr):
    """Requires 1D numpy array"""
    return (arr - arr.min()) / (arr.max() - arr.min())

def quantize(x, ratio=0.38):
    if x > ratio:
        return 1
    elif x < -ratio:
        return -1
    else:
        return 0
