"""
Ugur Gudelek
config
ugurgudelek
08-Mar-18
finance-cnn
"""



class Config:
    """

    """

    def __init__(self):
        self.stocks_dir = '../input/raw_data'
        self.stock_names = ['spy']
        self.label_after = 20

        self.input_size = 28
        self.seq_length = 28
        self.num_layers = 1
        self.out_size = 1

        self.train_batch_size = 100
        self.valid_batch_size = 100

        self.train_shuffle = True
        self.valid_shuffle = False

        self.epoch_size = 20
        self.storage_names = ['y_hat', 'loss', 'y']


def main():
    pass


if __name__ == "__main__":
    main()