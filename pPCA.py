import numpy as np


class pPCA:

    def __init__(self, latent_dim: int=50):
        """
        Initialize a probabilistic PCA model with the given latent dimension
        :param latent_dim: an int depicting the latent dimension the model should learn
        """
        self.latent = latent_dim
        self.mu, self.W, self.phi = np.array([]), np.array([]), np.array([])
        self.M, self.M_inv = np.array([]), np.array([])
        self._shape, self._d = [], 0
        self._trained = False
        self.__prec = 10e-4

    def __str__(self): return 'pPCA_z{}'.format(self.latent)

    def __repr__(self): return 'pPCA_z{}'.format(self.latent)

    def _update_M(self):
        """
        Updates inner parameters (which shouldn't be needed by anyone outside of the class)
        """
        self.M = self.W.T @ self.W
        self.M.flat[::self.M_inv.shape[0] + 1] += self.phi + self.__prec
        self.M_inv = np.linalg.inv(self.M)

    def fit(self, X: np.ndarray, y=None, verbose: bool=True):
        """
        Fit the probabilistic PCA model to the given data.
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param y: not used (defaults to None) - a parameter to fit the API of a standard sklearn model
        :return: the fitted model
        """
        self._shape = list(X.shape[1:])
        if verbose:
            print('Fitting a pPCA model to {} samples with latent dimension {}'.format(X.shape[0], self.latent))
        self._d = np.prod(self._shape)
        if X.ndim > 2:
            X = X.copy().reshape((X.shape[0], np.prod(X.shape[1:])))
        self.mu = X.sum(axis=0)/X.shape[0]
        _, s, v = np.linalg.svd(X - self.mu, full_matrices=False)
        sig = (s ** 2) / X.shape[0]

        self.phi = np.maximum(np.sum(sig[self.latent:]) / (self._d - self.latent), self.__prec)
        self.W = v[:self.latent, :].T * np.sqrt(np.maximum(sig[:self.latent] - self.phi, self.__prec))[None, :]

        self._update_M()
        self._trained = True
        return self

    def fit_transform(self, X: np.ndarray, y=None):
        """
        Fit a probabilistic PCA model to the given data and return the encodings of the data
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param y: not used (defaults to None) - a parameter to fit the API of a standard sklearn model
        :return: the encoded data points of X as a [N, latent_dim] numpy array
        """
        self.fit(X)
        self._trained = True
        return self.encode(X)

    def encode(self, X: np.ndarray):
        """
        Encode the datapoints of X using the learned pPCA model. If the model wasn't trained before calling
        this step, it is trained on the data points of X (essentially the same as calling
        model.fit_transform(X))
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :return: the encoded data points of X as a [N, latent_dim] numpy array
        """
        if not self._trained:
            self._trained = True
            return self.fit_transform(X)
        Z = X.copy().reshape((X.shape[0], np.prod(X.shape[1:])))
        return (self.M_inv @ self.W.T @ (Z.T - self.mu[:, None])).T

    def decode(self, Z: np.ndarray):
        """
        Decode the latent vectors Z into the original space
        :param Z: a numpy array with dimensions [N, latent_dim] where N is the number of samples to decode
        :return: a numpy array with dimensions [N, ...] where '...' stand for the original dimensions of
                 the data (the model reshapes the decoded vectors to the shapes of the original data)
        """
        assert self._trained, "Model must be trained before trying to decode"
        inved = self.W.T @ self.W
        inved.flat[::inved.shape[0] + 1] += 10e-8
        inved = np.linalg.inv(inved)
        X = (self.W @ inved @ self.M @ Z.T).T + self.mu[None, :]
        return X.reshape(np.hstack([X.shape[0], self._shape]))

    def generate(self, N: int):
        """
        Generate data from the learned model
        :param N: number of samples to generate
        :return: a numpy array of dimension [N, ...] of generated samples (where '...' stands for
                 the original shape of the data the model was trained on)
        """
        assert self._trained, "Model must be trained before trying to generate data from it"
        return self.decode(np.random.randn(N, self.latent))

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """
        Measure the Mahalanobis distance between the model parameters (mu and W) and the given samples
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to measure distance from
        :return: a numpy array with shape (N,), representing the Mahalanobis distance of each sample from the pPCA
        """
        assert self._trained, "Model must be trained before calculating the Mahalanobis distance from it"
        if X.ndim > 2:
            X = X.copy().reshape((X.shape[0], np.prod(X.shape[1:])))
        self._update_M()
        meaned_X = X.T - self.mu[:, None]
        mahala_p1 = np.sum(meaned_X * meaned_X, axis=0) / self.phi
        mahala_p2 = self.W.T @ meaned_X
        mahala_p2 = ((self.M_inv @ mahala_p2) * mahala_p2).sum(axis=0) / (self.phi ** 2)
        return mahala_p1 - mahala_p2

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        assert self._trained, "Model must be trained before calculating log-likelihood"
        mahala = self.mahalanobis(X)
        sign, d = np.linalg.slogdet(self.M_inv)
        det = self._d * np.log(self.phi) - sign * d
        ll = -0.5*(self._d*np.log(2*np.pi) + det + mahala)
        return ll

    def save(self, path: str=None):
        """
        Saves a pPCA model
        :param path: the path to save the model to; if no path was given, the model is saved in the current
                     directory with a default name format "pPCA_z<latent_dim>.pkl". The model is saved
                     using pickle
        """
        assert self._trained, "Model must be trained before trying to save it"
        params = {'mu': self.mu.copy(),
                  'W': self.W.copy(),
                  'phi': self.phi.copy(),
                  'latent': self.latent,
                  'shape': self._shape.copy(),
                  'd': self._d}
        if path is None: path = str(self)
        np.savez(path, **params)
        print('Saved model as file {}'.format(path))

    @staticmethod
    def load(path: str):
        """
        Loads a pPCA model
        :param path: the path to load the model from
        :return: the model if loading was successful, otherwise raises an error
        """
        try:
            params = np.load(path)
            mod = pPCA(latent_dim=params['latent'])
            mod._d, mod._shape = params['d'], params['shape']
            mod.mu, mod.W, mod.phi = params['mu'], params['W'], params['phi']
        except:
            raise Exception('Either the path {} is not a saved pPCA model or '
                            'there was a problem reading the file.'.format(path))
        mod._update_M()
        mod._trained = True
        return mod
