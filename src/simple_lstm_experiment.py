__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from dataset.toy.toyrnndataset import ToyRNNDataset
from model.lstm import LSTMTagger
from torchvision import transforms

from trainer.classifier import ClassifierTrainer
import torch


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 7

    hyperparams = dict(lr=0.001, train_batch_size=1, test_batch_size=1, epoch=10)
    params = dict(log_interval=2)

    dataset = ToyRNNDataset()

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,
                       vocab_size=len(dataset.word_to_ix),
                       tagset_size=len(dataset.tag_to_ix))



    trainer = ClassifierTrainer(model=model, dataset=dataset, hyperparams=hyperparams, params=params)

    trainer.fit()

# trainer.predict(data=dataset.testset.data)
#
# trainer.score(data=dataset.testset.data, targets=dataset.testset.targets)
