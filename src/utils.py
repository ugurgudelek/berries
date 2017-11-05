"""Utility methods"""


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

    def average(self):
        return sum(self.container) / self.size

    def __str__(self):
        string = ""
        string += str(self.container)
        return string

    def __repr__(self):
        return self.__str__()
