class History:

    def __init__(self):
        self.train_loss = EpochContainer()
        self.validation_loss = EpochContainer()






class EpochContainer:
    def __init__(self):
        self._container = {}


    def to_list(self):
        return list(self._container.values())

    def to_dict(self):
        return self._container

    def last(self):
        return self.to_list()[-1]

    def __setitem__(self, epoch, value):
        self._container[epoch] = value

    def __getitem__(self, epoch):
        return self._container[epoch]

    def __len__(self):
        return self._container.__len__()

    def __str__(self):
        return str(self.to_list())

    def __repr__(self):
        return self.__str__()





