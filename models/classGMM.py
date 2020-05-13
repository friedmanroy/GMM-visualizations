import numpy as np
from models.pPCA import pPCA
from scipy.special import logsumexp
from typing import Union


_lab_type = Union[list, np.ndarray]


class pPCA_cGMM:

    def __init__(self, latent_dim: int=50):
        """
        Initialize a probabilistic PCA model with the given latent dimension
        :param latent_dim: an int depicting the latent dimension the model should learn
        """
        self.latent = latent_dim
        self.labels, self.mix, self.priors = [], [], []
        self._shape, self._d = [], 0
        self._trained = False
        self.__prec = 10e-6

    def __str__(self): return 'cGMM_z{}'.format(self.latent)

    def __repr__(self): return 'cGMM_z{}'.format(self.latent)

    def fit(self, X: np.ndarray, y: _lab_type, verbose: bool=True):
        """
        Fit the probabilistic PCA model to the given data.
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param y: labels of the classes for each sample in X
        :return: the fitted model
        """
        assert X.shape[0] == len(y), 'The same number of samples and labels must be supplied'
        self._shape = list(X.shape[1:])
        self._d = np.prod(self._shape)
        if self.latent > self._d: self.latent = int(self._d)
        if verbose:
            print('Fitting a class-GMM model to {} samples with latent dimension {}'.format(X.shape[0], self.latent))

        labs = np.sort(np.unique(y))
        self.labels = labs
        self.mix = np.maximum(self.__prec, np.array([np.sum(y == l)/X.shape[0] for l in labs]))
        self.mix /= np.sum(self.mix)
        self.priors = [i for i in range(len(labs))]
        for i, l in enumerate(labs):
            self.priors[i] = pPCA(self.latent).fit(X[y == l], verbose=False)
        self._trained = True
        return self

    def generate(self, N: int, return_labels: bool=False, label=None):
        """
        Generate data from the trained model
        :param N: the number of samples to generate
        :param return_labels: whether the model should return the class labels for the generated data or not
        :return: if return_labels is set to False, the return value will be a ndarray with shape [N, ...] of generated
                 samples. If return_labels is set to True, in addition to the first output a ndarray with shape [N,]
                 will be outputed, with the labels that match the generated samples
        """
        assert self._trained, "Model must be trained before trying to generate data from it"
        if label is None: chosen = np.random.choice(len(self.labels), N, replace=True, p=self.mix)
        else:
            assert label in self.labels
            ind = [i for i, l in enumerate(self.labels) if l==label][0]
            chosen = np.array([ind for _ in range(N)])
        labs = [self.labels[c] for c in chosen]
        gen = []
        for i in chosen:
            gen.append(self.priors[i].generate(1)[0])
        if return_labels: return np.array(gen, copy=False), labs
        return np.array(gen, copy=False)

    def log_likelihood(self, X: np.ndarray):
        assert self._trained, "Model must be trained before calculating log-likelihood"
        ll = np.zeros((X.shape[0], len(self.labels)))
        for i, l in enumerate(self.labels):
            ll[:, i] = np.log(self.mix[i]) + self.priors[i].log_likelihood(X)
        return logsumexp(ll, axis=1)

    def predict(self, X: np.ndarray):
        assert self._trained, "Model must be trained before predicting labels"
        inds = np.argmax(self.predict_log_proba(X), axis=1)
        return [self.labels[i] for i in inds]

    def predict_proba(self, X: np.ndarray):
        assert self._trained, "Model must be trained before predicting probabilities"
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X: np.ndarray):
        assert self._trained, "Model must be trained before predicting log-probabilities"
        ll = np.zeros((X.shape[0], len(self.labels)))
        for i in range(len(self.labels)):
            ll[:, i] = np.log(self.mix[i]) + self.priors[i].log_likelihood(X)
        return ll - logsumexp(ll, axis=1)[:, None]

    def score(self, X: np.ndarray, y: _lab_type):
        assert self._trained, "Model must be trained before calculating score"
        p = self.predict(X)
        score = 0
        for i in range(len(p)):
            if y[i] == p[i]: score += 1
        return score/len(p)


class cGMM:

    def __init__(self):
        """
        Initialize a probabilistic PCA model with the given latent dimension
        :param latent_dim: an int depicting the latent dimension the model should learn
        """
        self.labels, self.mix, self.k = [], [], 0
        self.mu, self.cov = np.array([]), np.array([])
        self._shape, self._d = [], 0
        self._trained = False
        self.__prec = 10e-6

    def __str__(self): return 'cGMM'

    def __repr__(self): return 'cGMM'

    def fit(self, X: np.ndarray, y: _lab_type, verbose: bool=True):
        """
        Fit the probabilistic PCA model to the given data.
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param y: labels of the classes for each sample in X
        :return: the fitted model
        """
        assert X.shape[0] == len(y), 'The same number of samples and labels must be supplied'
        self._shape = list(X.shape[1:])
        self._d = np.prod(self._shape)
        if verbose: print('Fitting a class-GMM model to {} samples'.format(X.shape[0]))

        labs = np.unique(y)
        self.labels = labs
        self.k = len(labs)
        self.mix = np.maximum(self.__prec, np.array([np.sum(y == l)/X.shape[0] for l in labs]))
        self.mix /= np.sum(self.mix)
        self.mu = np.zeros((self.k, self._d))
        self.cov = np.zeros((self.k, self._d, self._d))
        for i, l in enumerate(labs):
            self.mu[i] = np.mean(X[y == l], axis=0)
            self.cov[i] = (X[y == l].T - self.mu[i, :, None])@(X[y == l].T - self.mu[i, :, None]).T/X.shape[0]
            self.cov[i].flat[::self._d+1] += self.__prec
        self._trained = True
        return self

    def generate(self, N: int, return_labels: bool=False, label=None):
        """
        Generate data from the trained model
        :param N: the number of samples to generate
        :param return_labels: whether the model should return the class labels for the generated data or not
        :return: if return_labels is set to False, the return value will be a ndarray with shape [N, ...] of generated
                 samples. If return_labels is set to True, in addition to the first output a ndarray with shape [N,]
                 will be outputed, with the labels that match the generated samples
        """
        assert self._trained, "Model must be trained before trying to generate data from it"
        if label is None: chosen = np.random.choice(len(self.labels), N, replace=True, p=self.mix)
        else:
            assert label in self.labels
            ind = [i for i, l in enumerate(self.labels) if l==label][0]
            chosen = np.array([ind for _ in range(N)])
        labs = [self.labels[c] for c in chosen]
        chol = np.linalg.cholesky(self.cov)
        eps = np.random.randn(N, self._d, 1)
        gen = (chol[chosen]@eps.T)[:, :, 0] + self.mu[chosen]
        if return_labels: return gen, labs
        return gen

    def log_likelihood(self, X: np.ndarray):
        assert self._trained, "Model must be trained before calculating log-likelihood"
        return logsumexp(self.logpdf(X), axis=1)

    def predict(self, X: np.ndarray):
        assert self._trained, "Model must be trained before predicting labels"
        inds = np.argmax(self.predict_log_proba(X), axis=1)
        return [self.labels[i] for i in inds]

    def predict_proba(self, X: np.ndarray):
        assert self._trained, "Model must be trained before predicting probabilities"
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X: np.ndarray):
        assert self._trained, "Model must be trained before predicting log-probabilities"
        ll = self.logpdf(X)
        return ll - logsumexp(ll, axis=1)[:, None]

    def score(self, X: np.ndarray, y: _lab_type):
        assert self._trained, "Model must be trained before calculating score"
        p = self.predict(X)
        score = 0
        for i in range(len(p)):
            if y[i] == p[i]: score += 1
        return score/len(p)

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        assert self._trained, "Model must be trained before calculating log-pdf of all clusters"
        _, d = np.linalg.slogdet(self.cov)
        meaned = X[None, :] - self.mu[:, None, :]
        inved = np.linalg.inv(self.cov)
        return (np.log(self.mix[:, None]) - .5*(d[:, None] + np.sum(meaned * (meaned @ inved), axis=2))).T
