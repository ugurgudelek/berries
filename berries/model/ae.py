import torch
from torch import nn
from torch.nn import functional as F


class AE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(AE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = input_dim

        self.encoder = Encoder(input_dim=self.input_dim,
                               latent_dim=self.latent_dim)
        self.decoder = Decoder(latent_dim=self.latent_dim,
                               output_dim=self.output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        z = self.encoder(x)
        return self.decoder(z)

    def latent(self, x):
        x = x.view(-1, self.input_dim)
        return self.encoder(x)


class Encoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2), nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.input_dim // 4), nn.ReLU(),
            nn.Linear(self.input_dim // 4, self.latent_dim))

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.output_dim // 4), nn.ReLU(),
            nn.Linear(self.output_dim // 4, self.output_dim // 2), nn.ReLU(),
            nn.Linear(self.output_dim // 2, self.output_dim))

    def forward(self, x):
        return self.decoder(x)
