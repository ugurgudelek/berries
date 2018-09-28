from collections import defaultdict
import numpy as np

class History:
    """
        Args:
            what_to_store (list): list of labels for what to store.
    """

    def __init__(self, epoch_size, storage_names):

        self.storage_names = storage_names

        # container -> epoch_num -> phase -> what_to_store

        # init for epoch_num
        self.container = {i: {} for i in range(epoch_size)}

        # init for phase
        for epoch in self.container.keys():
            self.container[epoch]['train'] = {}
            self.container[epoch]['valid'] = {}

            # init for what_to_store
            for w in storage_names:
                self.container[epoch]['train'][w] = np.array([])
                self.container[epoch]['valid'][w] = np.array([])


    def append(self, epoch, phase, name, value):
        """
        """
        if name not in self.storage_names:
            raise Exception('key:{} not available in history'.format(name))

        data = self.container[epoch][phase][name]
        self.container[epoch][phase][name] = np.append(data, value)


    def set(self, epoch, phase, name, value):
        """
        """
        if name not in self.storage_names:
            raise Exception('key:{} not available in history'.format(name))

        self.container[epoch][phase][name] = value

    def get(self, epoch, phase, name):
        """
        """
        if name not in self.storage_names:
            raise Exception('key:{} not available in history'.format(name))

        return self.container[epoch][phase][name]

#    def last(self, label):
#         """
#
#         Args:
#             label:
#
#         Returns:
#         Raises: KeyError
#         """
#         arr = self.container[label]
#         if len(arr) == 0:
#             return np.inf
#
#         return arr[-1]
