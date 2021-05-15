from sklearn.decomposition import PCA as skPCA


class PCA():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.pca = skPCA(self.latent_dim)

    def forward(self, x):
        z = self.transform(x)
        recon = self.inverse_transform(z)
        return recon

    def transform(self, x):
        return self.pca.transform(x)

    def inverse_transform(self, z):
        return self.pca.inverse_transform(z)

    def fit(self, x):
        self.pca.fit(x)
