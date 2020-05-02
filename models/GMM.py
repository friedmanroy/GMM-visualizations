import numpy as np
from scipy.special import logsumexp
import tqdm


class GMM:

    def __init__(self, k: int=10):
        """
        Initialize a probabilistic PCA model with the given latent dimension
        :param latent_dim: an int depicting the latent dimension the model should learn
        """
        self.k = k
        self.mix = np.ones(k)/k
        self.mu = np.array([])
        self.cov = np.array([])
        self._d = 0
        self._trained = False
        self._inited = False
        self.__prec = 10e-6

    def __str__(self): return 'GMM_k{}'

    def __repr__(self): return 'GMM_k{}'

    def __check_X(self, X: np.ndarray):
        assert X.ndim == 2
        assert X.shape[1] == self._d

    def responsibilities(self, X: np.ndarray, log: bool=False) -> np.ndarray:
        assert self._inited
        self.__check_X(X)
        res = np.zeros((self.k, X.shape[0]))
        for k in range(self.k):
            res[k, :] = np.log(self.mix[k]) - 0.5*(self.mahalanobis(X, k) + np.linalg.slogdet(self.cov[k])[1])
        res = res -logsumexp(res, axis=0)[None, :]
        return res if log else np.exp(res)

    def mahalanobis(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Calculates the Mahalanobis distance of the samples X from the cluster k
        :param X: the data to calculate the distance from, as a numpy array with shape [# samples, # features]
        :param k: the cluster index
        :return: a numpy array with shape [# samples] of the distances of each sample from the cluster
        """
        assert self._inited
        self.__check_X(X)
        meaned = X - self.mu[k]
        return np.sum(meaned*(meaned @ np.linalg.inv(self.cov[k])), axis=1)

    def fit(self, X: np.ndarray, iterations: int=10, verbose: bool=True):
        """
        Fit a GMM to the given datapoints
        :param X: the data to fit to, as a numpy array with shape [# samples, # features]
        :param iterations: the number of iterations to fit the model to the data for
        :param verbose: whether progress should be printed or not
        :return: the trained model
        """
        if not self._inited:
            assert X.ndim == 2
            self._d = X.shape[1]
            self.mu = np.zeros((self.k, self._d))
            for k in range(self.k):
                self.mu[k] = np.mean(X[np.random.choice(X.shape[0], int(X.shape[0]/self.k), replace=False)], axis=0)
            self.cov = np.zeros((self.k, self._d, self._d))
            self.cov += .5*np.eye(self._d)[None, ...]
            self._inited = True
        else: self.__check_X(X)

        if verbose: print('Fitting a class-GMM model to {} samples with {} clusters, for {} iterations:'
                          .format(X.shape[0], self.k, iterations), flush=True)
        for _ in tqdm.tqdm(range(iterations), disable=not verbose):
            res = self.responsibilities(X, log=True)
            norm = np.exp(logsumexp(res, axis=1))
            res = np.exp(res)

            self.mix = np.maximum(np.sum(res, axis=1)/X.shape[0], self.__prec)
            self.mix = self.mix / np.sum(self.mix)
            for k in range(self.k):
                self.mu[k] = np.sum(res[k, :, None]*X, axis=0)/norm[k]
                self.cov[k] = ((X-self.mu[k]).T @ (res[k, :, None]*(X-self.mu[k]))) / norm[None, k]
                self.cov[k].flat[::self.cov[k].shape[0]+1] += self.__prec

        self._trained = True
        return self

    def generate(self, N: int, return_labels: bool=False, clust: int=None):
        """
        Generate data from the trained model
        :param N: the number of samples to generate
        :param return_labels: whether the model should return the class labels for the generated data or not
        :param clust: the cluster to generate the samples from. If no cluster is supplied, each generated data point
                      will be sampled according to the mixing probabilities
        :return: if return_labels is set to False, the return value will be a ndarray with shape [N, ...] of generated
                 samples. If return_labels is set to True, in addition to the first output a ndarray with shape [N,]
                 will be outputed, with the labels that match the generated samples
        """
        assert self._trained, "Model must be trained before trying to generate data from it"
        if clust is None: chosen = np.random.choice(self.k, N, replace=True, p=self.mix)
        else:
            assert clust < self.k
            chosen = np.ones(N)*clust
        labs = [c for c in chosen]
        gen = []
        for c in chosen:
            gen.append(np.random.multivariate_normal(mean=self.mu[c], cov=self.cov[c], size=1)[0])
        if return_labels: return np.array(gen, copy=False), labs
        return np.array(gen, copy=False)

    def log_likelihood(self, X: np.ndarray):
        self.__check_X(X)
        assert self._trained, "Model must be trained before calculating log-likelihood"
        ll = np.zeros((self.k, X.shape[0]))
        for k in range(self.k):
            ll[k, :] = np.log(self.mix[k]) - 0.5*(self.mahalanobis(X, k) + np.linalg.slogdet(self.cov[k])[1])
        return logsumexp(ll, axis=0)

    def predict(self, X: np.ndarray):
        self.__check_X(X)
        assert self._trained, "Model must be trained before predicting labels"
        inds = np.argmax(self.predict_log_proba(X), axis=0)
        return [i for i in inds]

    def predict_proba(self, X: np.ndarray):
        self.__check_X(X)
        assert self._trained, "Model must be trained before predicting probabilities"
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X: np.ndarray):
        self.__check_X(X)
        assert self._trained, "Model must be trained before predicting log-probabilities"
        return self.responsibilities(X, log=True)
