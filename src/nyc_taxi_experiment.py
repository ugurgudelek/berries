import torch
import torch.nn as nn
from dataset.nyctaxi import *
import matplotlib.pyplot as plt
from model.genericrnn import RNNPredictor
from torch.optim import Adam

from trainer.rnntrainer import RNNTrainer
from dataset.generic import Standardizer, TimeSeriesDatasetWrapper

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    hyperparams = {
        # Model params
        'model': 'LSTM',
        'emsize':32,
        'nhid': 32,
        'nlayers': 2,
        'res_connection': False,  # resudial connections
        'dropout': 0.2,
        'tied': False,  # tie the word embedding and softmax weights (deprecated)
        # Optim params
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'clip': 10,  # gradient clipping
        # Dataloader params
        'train_batch_size': 64,
        'test_batch_size': 64,
        'seq_len': 50,  # sequence length

        'teacher_forcing_ratio': 0.7,  # teacher forcing ratio (deprecated)
        'epoch': 1000,  # upper epoch limits
    }
    params = {'log_interval': 10,
              'save_interval': 10,
              'save_fig': True,
              'resume': False,
              'pretrained': False,
              'prediction_window_size': 10,
              'augment': True,
              'seed': 42,
              'device': 'cuda',
              'experiment_name': 'nyc_taxi'
              }

    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    scaler = Standardizer()

    train_dataset = NYCTaxiDataset.from_pickle(train=True, scaler=scaler, augment=params['augment'])
    test_dataset = NYCTaxiDataset.from_pickle(train=False, scaler=scaler)

    dataset = TimeSeriesDatasetWrapper(trainset=train_dataset,
                                    testset=test_dataset)


    model = RNNPredictor(rnn_type=hyperparams['model'],
                         enc_inp_size=dataset.feature_dim,
                         rnn_inp_size=hyperparams['emsize'],
                         rnn_hid_size=hyperparams['nhid'],
                         dec_out_size=dataset.feature_dim,
                         nlayers=hyperparams['nlayers'],
                         dropout=hyperparams['dropout'],
                         tie_weights=hyperparams['tied'],
                         res_connection=hyperparams['res_connection']).to(params['device'])
    trainer = RNNTrainer(model=model,
                               dataset=dataset,
                               hyperparams=hyperparams,
                               params=params)

    trainer.fit()


