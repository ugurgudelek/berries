import torch
import torch.utils.data
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm
import time
import random

from config import *
import plots
from history import History


class Experiment:

    def __init__(self, experiment_dir, model, dataset_params, device, classification,
                 sequence_sample, predict_n_step, future_predict_initial_step_size, verbose=10, seed=42):

        self.seed_random(seed=seed)

        self.model = model
        self.dataset = dataset_params['dataset']
        self.device = device
        self.use_cuda = False if self.device.type == 'cpu' else True
        self.experiment_dir = experiment_dir
        self.classification = classification

        self.sequence_sample = sequence_sample
        self.predict_n_step = predict_n_step
        self.future_predict_initial_step_size = future_predict_initial_step_size
        self.verbose = verbose

        self.epoch = 0

        os.makedirs(os.path.join(self.experiment_dir, 'checkpoints'), exist_ok=True)

        # generator for training dataset
        self.train_dataloader = DataLoader(self.dataset.train_dataset,
                                           batch_size=dataset_params['train_batch_size'],
                                           shuffle=False,
                                           sampler=dataset_params.get('sampler', None),
                                           drop_last=True,
                                           pin_memory=self.use_cuda)  # allocate the samples in page-locked memory, which speeds-up the transfer

        # generator for validation dataset
        self.valid_dataloader = DataLoader(self.dataset.valid_dataset,
                                           batch_size=dataset_params['valid_batch_size'],
                                           shuffle=False,
                                           sampler=None,
                                           drop_last=True,
                                           pin_memory=self.use_cuda)

        # tensorboard summary writer
        # self.writer = SummaryWriter(log_dir=os.path.join(self.config.EXPERIMENT_DIR, 'summary'))

        # save config and model
        self.save_init()

        self.plotter = plots.Plotter(title=dataset_params['dataset_name'], path=self.experiment_dir)
        self.history = History()

    @classmethod
    def maybe_resume(cls, epoch=None, *args, **kwargs):


        # if not os.path.exists(experiment_dir): # if resume_dir is not exist
        #     return cls(*args, **kwargs)

        # cls.experiment_dir = experiment_dir
        # change new experiment directory to existing one.

        experiment_dir = kwargs['experiment_dir']
        experiment = cls(*args, **kwargs)
        checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')

        if os.listdir(checkpoints_dir).__len__() != 0:
            if epoch is None:  # resume from last saved checkpoint
                epochs = sorted([int(os.path.splitext(l)[0]) for l in os.listdir(checkpoints_dir)])
                epoch = epochs[-1]
            # else: resume from given checkpoint
            experiment.load_experiment(experiment_dir, epoch)

        # else: start new experiment
        return experiment

    def save_experiment(self, directory):

        os.makedirs(os.path.join(directory, 'checkpoints', str(self.epoch)), exist_ok=True)
        # Save AUX data
        torch.save({'epoch': self.epoch,
                    'train_loss': self.history.train_loss,
                    'validation_loss': self.history.validation_loss},
                   os.path.join(directory, 'checkpoints', str(self.epoch), "aux-info.tar"))
        # Save Model and Optimizer
        self.model.save_checkpoint(directory, self.epoch)

    def load_experiment(self, directory, epoch):
        # Load AUX data
        checkpoint_aux = torch.load(os.path.join(directory, 'checkpoints', str(epoch), "aux-info.tar"))
        self.history.train_loss = checkpoint_aux['train_loss']
        self.history.validation_loss = checkpoint_aux['validation_loss']
        self.epoch = checkpoint_aux['epoch']

        # Load Model and Optimizer
        self.model.load_checkpoint(directory, epoch)

    def save_init(self):
        """Saves the config file and model structure"""
        # create experiment_dir and save .ini file
        os.makedirs(self.experiment_dir, exist_ok=True)
        # self.config.save()

        # save model layers to txt and onnx
        # self.model.save(directory=self.config.EXPERIMENT_DIR)

    def run_epoch(self, pbar):

        self.model.reset_states()
        # ================ FIT ================
        tloss = []
        for step, (X, y) in enumerate(self.train_dataloader):
            # Fit the model
            self.model.fit(X.to(self.device).float(),
                           y.to(self.device).float())
            tloss += [self.model.train_loss]
        self.history.train_loss[self.epoch] = np.array(tloss).mean()

        # ================ VALIDATION ================
        vloss = []
        for step, (X, y) in enumerate(self.valid_dataloader):
            # Validate on validation set
            self.model.validate(X.to(self.device).float(),
                                y.to(self.device).float())
            # todo: current build call .validate when .score is used!
            vloss += [self.model.validation_loss]
        self.history.validation_loss[self.epoch] = np.array(vloss).mean()

        # ================ LOG ================
        if ((self.epoch + 1) % self.verbose) == 0:
            self.plotter.learning_curve.add_vector(name='train_loss', y=self.history.train_loss.to_list())
            self.plotter.learning_curve.add_vector(name='validation_loss', y=self.history.validation_loss.to_list())


            # ================ PREDICTION ================

            X_batch, y_batch = self.dataset.sampler.get_data(dataset=self.dataset.train_dataset, n=self.predict_n_step)
            self.plotter.predicton_curve.add_vector(name='y', y=y_batch, linewidth=4)

            X_valid_batch, y_valid_batch = self.dataset.sampler.get_data(dataset=self.dataset.valid_dataset, n=self.predict_n_step)
            self.plotter.predicton_curve.add_vector(name='y_valid', y=y_valid_batch)

            # # region Train LSTM Sequences
            # lstm_output_batch = self.model.predict_one_step_ahead(X_batch=X_batch,
            #                                                 classification=False,
            #                                                 return_sequences=True)
            # for cell_num in range(lstm_output_batch.shape[1]):  # foreach hidden
            #     self.plotter.lstminner_curve.add_vector(name=f'lstm_out_{cell_num}', y=lstm_output_batch[:, cell_num])
            # # endregion

            # region Valid LSTM Sequences
            lstm_output_batch_valid = self.model.predict_one_step_ahead(X_batch=X_valid_batch,
                                                            classification=False,
                                                            return_sequences=True)
            for cell_num in range(lstm_output_batch_valid.shape[1]):  # foreach hidden
                self.plotter.lstminner_curve.add_vector(name=f'lstm_out_{cell_num}', y=lstm_output_batch_valid[:, cell_num])
            # endregion

            # region  Test Prediction By Hand
            # fc_weight = self.model.state_dict()['fc.0.weight'].cpu().data.numpy()
            # fc_bias = self.model.state_dict()['fc.0.bias'].cpu().data.numpy()
            # by_hand_batch = np.dot(lstm_output_batch, fc_weight.T) + fc_bias
            # self.plotter.predicton_curve.add_vector(name='by_hand', y=by_hand_batch[:, 0])
            # endregion

            # region Future Prediction
            future_predictions = self.model.predict_future(initial_x=X_batch[:self.future_predict_initial_step_size, :, :],
                                                           n_steps=self.predict_n_step,  # actual n_steps: n_steps-initial_x.shape[0]
                                                           classification=self.classification)
            self.plotter.predicton_curve.add_vector(name='future', y=future_predictions[:, 0])
            # endregion

            # region Train Prediction
            predictions_batch = self.model.predict_one_step_ahead(X_batch=X_batch,
                                                                  classification=self.classification,
                                                                  return_sequences=False)

            self.plotter.predicton_curve.add_vector(name='prediction', y=predictions_batch[:, 0], dash='dot')
            # endregion

            # region Test Prediction
            predictions_valid_batch = self.model.predict_one_step_ahead(X_batch=X_valid_batch,
                                                                  classification=self.classification,
                                                                  return_sequences=False)

            self.plotter.predicton_curve.add_vector(name='prediction_valid', y=predictions_valid_batch[:, 0])

            # endregion

            self.plotter.update()

            print(f"\n===============\n{self.model.state_dict()}\n===============\n")

            self.save_experiment(directory=self.experiment_dir)
            self.load_experiment(directory=self.experiment_dir, epoch=self.epoch)

            # Write losses to the tensorboard
            # self.writer.add_scalar('training_loss', train_loss, epoch)
            # self.writer.add_scalar('validation_loss', validation_loss, epoch)



            # Write PR Curve to the summary writer.
            # self.writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), epoch)

            # for name, param in model.named_parameters():
            #     print(name)
            #     print(param)
            #     model.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch, bins=100)
            # x = dict(model.named_parameters())['conv1.weight'].clone().cpu().data.numpy()
            # kernel1= x[0,0]
            # plt.imshow(kernel1)
            # plt.show()
            # needs tensorboard 0.4RC or later


        self.update_tqdm(pbar=pbar,
                         tloss=self.history.train_loss.last(),
                         vloss=self.history.validation_loss.last())

    def update_tqdm(self, pbar, tloss, vloss, precision=10):
        # f'{value:{width}.{precision}}'
        pbar.set_description(f"Eoch:{self.epoch}|TLoss:{tloss:.{precision}f}|VLoss:{vloss:.{precision}f}")
        pbar.update(1)

    def seed_random(self, seed=7):

        random.seed(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed(seed=seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(np.random.rand())

    # def validate_before_run(self):
    #     validation_loss = []
    #     for step, (X, y) in enumerate(self.valid_dataloader):
    #         print(step,': ',X.item(),'->',y.item(), end=' ')
    #         if (step % self.config.HIDDEN_RESET_PERIOD) == 0:
    #             self.model.init_hidden(inplace=True)
    #             print(" -> Hiddens initialized.")
    #         else:
    #             print(" -> No init")
    #
    #         self.model.validate(X.to(self.config.DEVICE).float(), y.to(self.config.DEVICE).float())
    #         validation_loss += [self.model.validation_loss]
    #
    #     # X_sample, y_sample = self.dataset.random_train_sample(n=self.config.TRAIN_BATCH_SIZE)
    #     # X_sample, y_sample = X_sample.contiguous(), y_sample.contiguous()
    #     # predicted_labels = self.model.predict(X_sample.to(self.config.DEVICE)).cpu().detach()
    #     #
    #     # X_sample = self.dataset.denormalize(X_sample.data.numpy())
    #     # y_sample = self.dataset.denormalize(y_sample.data.numpy())
    #     # predicted_labels = self.dataset.denormalize(predicted_labels.data.numpy())
    #     # # print(X_sample.flatten(), y_sample, predicted_labels)
    #     # for x, y, p in zip(X_sample, y_sample, predicted_labels):
    #     #     print(f"{list(x.flatten())}->{y} but {p.item()}")
    #
    #     print(f"Validation Loss Before Run: {np.array(validation_loss).mean()}")

    def run(self, epoch_size):
        with tqdm(total=epoch_size, initial=self.epoch) as pbar:
            for self.epoch in range(self.epoch, epoch_size):
                self.run_epoch(pbar=pbar)
                if self.history.validation_loss.last() < 10e-8:
                    print(f"Validation loss {self.history.validation_loss.last()} < 10e-8. Training is finalized. \\m/")
                    break

        # self.writer.export_scalars_to_json(self.config.EXPERIMENT_DIR + '.json')
