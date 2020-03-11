__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

class Trainer:
    def __init__(self):
        pass

    def save_checkpoint(self, epoch):
        raise NotImplementedError()

    def load_checkpoint(self, epoch):
        raise NotImplementedError()

    def fit(self):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

    def proba(self):
        raise NotImplementedError()

    def predict_log_proba(self):
        raise NotImplementedError()

    def predict_proba(self):
        raise NotImplementedError()