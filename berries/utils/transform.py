__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"


class Normalizer():
    def __init__(self):
        pass

    def fit(self, data):
        self.min = data.min()
        self.max = data.max()
        return self

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * (self.max - self.min)) + self.min


class Standardizer():
    def __init__(self):
        pass

    def fit(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
