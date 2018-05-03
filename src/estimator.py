from model import CNN, LSTM
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torch.autograd import Variable
from torch import FloatTensor
import numpy as np


from tensorboardX import SummaryWriter


# class LoadEstimator:
#     """
#     todo: Please add docstring
#     """
#
#     # TODO: save experiment settings
#
#     def __init__(self, config, resume=False):
#         """
#
#         Args:
#             config:
#         """
#         self.config = config
#         # if we seed random func, they will generate same output everytime.
#         if config.RANDOM_SEED is not None:
#             torch.manual_seed(config.RANDOM_SEED)
#             np.random.seed(config.RANDOM_SEED)
#
#         dataset = LoadFullDataset(csv_path=config.INPUT_PATH,
#                                   train_valid_ratio=config.TRAIN_VALID_RATIO,
#                                   train_day=config.TRAIN_DAY,
#                                   seq_length=config.SEQ_LENGTH)
#
#         self.train_dataset = dataset.train_dataset
#         self.valid_dataset = dataset.valid_dataset
#
#         self.train_dataloader = DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, drop_last=True)
#         self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=config.BATCH_SIZE, drop_last=True)
#
#         self.model = LoadLSTM(input_size=config.INPUT_SIZE,
#                               seq_length=config.SEQ_LENGTH,
#                               num_layers=config.NUM_LAYERS,
#                               batch_size=config.BATCH_SIZE)
#
#         self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)
#         self.history = History(what_to_store=['train_loss', 'valid_loss', 'test_loss'])
#         self.plotter = Plotter(xlim=(0, config.SEQ_LENGTH), ylim=(0, 1), block=False)
#
#         self.experiment_dir = config.EXPERIMENT_DIR
#         self.epoch = 0
#
#         if resume:
#             self.load_from_latest_ckpt()
#
#     def load_from_latest_ckpt(self):
#         """
#
#         Returns:
#
#         """
#         latest_ckpt_path = Checkpoint.get_latest_checkpoint(self.experiment_dir)
#
#         print("model reading from {} ...".format(latest_ckpt_path))
#
#         latest_ckpt = Checkpoint.load(path=latest_ckpt_path)
#
#         self.model = latest_ckpt.model
#         self.optimizer = latest_ckpt.optimizer
#         self.epoch = latest_ckpt.epoch + 1  # increment by 1 to train next epoch
#         self.history = latest_ckpt.history
#
#     def _train_on_batch(self, X_batch, y_batch):
#         """
#
#         Args:
#             X_batch:
#             y_batch:
#
#         Returns:
#
#         """
#
#         self.optimizer.zero_grad()  # pytorch accumulates gradients.
#         gc.collect()
#         self.model.hidden = self.model.init_hidden()  # detach history of initial hidden
#         lstm_out, hidden = self.model(X_batch)
#
#         # prediction = hidden[0][-1, :, :]
#         prediction = lstm_out[-1, :, -1]
#         loss = self.criterion(prediction, y_batch)
#
#         loss.backward()
#         self.optimizer.step()
#
#         return lstm_out, hidden, prediction, loss
#
#     def _train_on_epoch(self, epoch):
#         """
#
#         Args:
#             epoch:
#
#         Returns:
#
#         """
#         self.model.train(mode=True)
#
#         for batch_num, (X, y) in enumerate(self.train_dataloader):
#             batch_size = X.size()[1]
#             step = batch_size * batch_num
#
#             (X, y) = Variable(X.float(), requires_grad=False), Variable(y.float(), requires_grad=False)
#             (lstm_out, hidden, prediction, train_loss) = self._train_on_batch(X_batch=X, y_batch=y)
#
#             self.history.append(label='train_loss', value=train_loss.data.numpy()[0].item())
#
#             print("epoch : {:>8} || batch_num : ({:>4}/{:<4}) || train_loss : {:.5f} || valid_loss  {:.5f}".format(
#                 epoch, batch_num, len(self.train_dataloader), self.history.last('train_loss'),
#                 self.history.last('valid_loss')))
#
#             if True:  # (batch_num + 1) % 10 == 0:
#                 X_to_plot = X.data.numpy()[0, :, 0]
#                 y_to_plot = y.data.numpy()[0]
#                 prediction_to_plot = prediction.data.numpy()[0]
#                 self.plotter.add(what_to_plot=X_to_plot, plot_type='line', label='X')
#                 self.plotter.add(what_to_plot=y_to_plot, plot_type='line', label='true')
#                 self.plotter.add(what_to_plot=prediction_to_plot, plot_type='line', label='pred')
#                 self.plotter.add(what_to_plot=self.history.get('train_loss'), plot_type='line', label='train_loss')
#                 self.plotter.add(what_to_plot=self.history.get('valid_loss'), plot_type='line', label='valid_loss')
#                 self.plotter.plot()
#
#         # save model, optimizer, epoch, history to the experiment_dir/datetime_epoch
#         Checkpoint(model=self.model, optimizer=self.optimizer,
#                    epoch=self.epoch, history=self.history,
#                    experiment_dir=self.experiment_dir).save()
#
#     def _validate(self):
#         # TODO: Implement self._validate and append loss to the history container
#         self.model.eval()
#         valid_losses = []
#         for batch_num, (X, y) in enumerate(self.valid_dataloader):
#             (X, y) = Variable(X.float(), requires_grad=False), Variable(y.float(), requires_grad=False)
#             self.model.hidden = self.model.init_hidden()
#             lstm_out, hidden = self.model(X)
#
#             # prediction = hidden[0][-1, :, :]
#             prediction = lstm_out[-1, :, -1]
#             valid_loss = self.criterion(prediction, y)
#
#             valid_losses.append(valid_loss.data.numpy()[0].item())
#
#         self.history.append(label='valid_loss', value=sum(valid_losses) / len(valid_losses))
#
#     def test(self):
#         # TODO: Implement self._test and append loss to the history container
#         self.model.eval()
#         valid_losses = []
#         predictions = []
#         Xs = []
#         ys = []
#         for batch_num, (X, y) in enumerate(self.valid_dataloader):
#             (X, y) = Variable(X.float(), requires_grad=False), Variable(y.float(), requires_grad=False)
#             self.model.hidden = self.model.init_hidden()
#             lstm_out, hidden = self.model(X)
#
#             # prediction = hidden[0][-1, :, :]
#             prediction = lstm_out[-1, :, -1]
#             valid_loss = self.criterion(prediction, y)
#
#             valid_losses.append(valid_loss.data.numpy()[0].item())
#
#             Xs.append(X)
#             ys.append(y)
#             predictions.append(prediction.data.numpy()[0].item())
#         # self.history.append(label='test_loss', value=test_loss.data.numpy()[0])
#         return {'Xs': Xs, 'ys': ys, 'predictions': predictions, 'valid_losses': valid_losses}
#
#     def train(self, epoch_size=20):
#         """
#         Iterates from latest epoch to epoch_size because maybe model is resuming from latest checkpoint.
#         Updates self.epoch every time too to be ready for next saving process.
#         Args:
#             epoch_size: how many epoch do we want to train the model?
#
#         Returns:
#
#         """
#         for self.epoch in range(self.epoch, epoch_size):
#             self._train_on_epoch(epoch=self.epoch)
#             self._validate()
#             # self._test()
class Estimator:
    """

    """

    def __init__(self, dataset, model_config, dataloader_config):

        self.model = LSTM(input_size=model_config['input_size'],
                          seq_length=model_config['seq_length'],
                          num_layers=model_config['num_layers'],
                          out_size=model_config['out_size'],
                          batch_size=model_config['batch_size'])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset.train_dataset,
                                           batch_size=dataloader_config['train_batch_size'],
                                           shuffle=dataloader_config['train_shuffle'],
                                           drop_last=True)
        self.valid_dataloader = DataLoader(dataset.valid_dataset,
                                           batch_size=dataloader_config['valid_batch_size'],
                                           shuffle=dataloader_config['valid_shuffle'],
                                           drop_last=True)

        self.writer = SummaryWriter()

        dummy_input = Variable(torch.rand(13, 1, 28, 28))

        self.writer.add_graph(self.model, (dummy_input,))

    def run_epoch(self, epoch):

        # Train
        toutputs, tlosses = np.array([]), np.array([])
        for step, (tX, ty) in enumerate(self.train_dataloader):
            print('step : {}'.format(step))
            tX, ty = Variable(tX.float(), requires_grad=False), Variable(ty.float(), requires_grad=False)

            toutput, tloss = self.train_on_batch(tX, ty)

            toutputs = np.concatenate((toutputs, toutput.data.numpy()),
                                      axis=0) if toutputs.size else toutput.data.numpy()
            tlosses = np.concatenate((tlosses, tloss.data.numpy()), axis=0) if tlosses.size else tloss.data.numpy()

        epoch_training_loss = tlosses.mean()

        # Validate
        voutputs, vlosses = np.array([]), np.array([])
        for i, (vX, vy) in enumerate(self.valid_dataloader):
            vX, vy = Variable(vX.float(), requires_grad=False), Variable(vy.float(), requires_grad=False)
            voutput, vloss = self.validate_on_batch(vX, vy)

            voutputs = np.concatenate((voutputs, voutput.data.numpy()), axis=0) if voutputs.size else voutput.data.numpy()
            vlosses = np.concatenate((vlosses, vloss.data.numpy()), axis=0) if vlosses.size else vloss.data.numpy()

        epoch_validation_loss = vlosses.mean()


        self.writer.add_scalar('training_loss', epoch_training_loss , epoch)
        self.writer.add_scalar('validation_loss', epoch_validation_loss, epoch)

        return (toutputs, epoch_training_loss, voutputs, epoch_validation_loss)

    def train_on_batch(self, Xs, ys):
        self.optimizer.zero_grad()  # pytorch accumulates gradients.

        # forward + backward + optimize
        self.model.hidden = self.model.init_hidden()  # detach history of initial hidden
        output = self.model(Xs)
        loss = self.criterion(output, ys)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return output, loss

    def validate_on_batch(self, Xs, ys):
        self.model.eval()

        output, loss = self.train_on_batch(Xs, ys)

        self.model.train()

        return output, loss

    def predict(self, Xs):
        self.model.eval()

        self.model.hidden = self.model.init_hidden(batch_size=1)
        pX = Variable(FloatTensor(Xs), requires_grad=False)
        output = self.model(pX)

        self.model.train()

        return output.data.numpy()

