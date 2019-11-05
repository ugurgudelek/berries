__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.utils.data import Dataset
from dataset.generic import GenericDataset


class ToyRNNDataset(GenericDataset):

    def __init__(self):
        training_data = [
            ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
            ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
        ]

        self.word_to_ix = {}
        for sent, tag in training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

        self.tag_to_ix = {'DET': 0, 'NN': 1, 'V': 2}



        self.trainset = InnerToyRNNDataset(data=[self.prepare_sequence(sent, self.word_to_ix)
                                                 for sent, tag in training_data],
                                           targets=[self.prepare_sequence(tag, self.tag_to_ix)
                                                 for sent, tag in training_data])

        self.testset = InnerToyRNNDataset(data=[self.prepare_sequence(sent, self.word_to_ix)
                                                 for sent, tag in training_data],
                                           targets=[self.prepare_sequence(tag, self.tag_to_ix)
                                                 for sent, tag in training_data])

    @staticmethod
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)


class InnerToyRNNDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, ix):
        return self.data[ix], self.targets[ix]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    toyrnn_dataset = ToyRNNDataset()
    print(ToyRNNDataset.prepare_sequence("The dog ate the apple".split(),
                                         toyrnn_dataset.word_to_ix))
    print(ToyRNNDataset.prepare_sequence(["DET", "NN", "V", "DET", "NN"],
                                         toyrnn_dataset.tag_to_ix))
