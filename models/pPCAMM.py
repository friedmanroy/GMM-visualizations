import numpy as np
from scipy.special import logsumexp
import tqdm
from typing import Union
from models.pPCA import pPCA


class GMM:

    def __init__(self, k: int=10, latent_dim: int=10):
        """
        Initialize a probabilistic PCA model with the given latent dimension
        :param latent_dim: an int depicting the latent dimension the model should learn
        """
        self.k = k
        self.q = latent_dim
        self.mix = np.ones(k)/k
        self.mu = np.array([])
        self.W = np.array([])
        self.phi = np.array([])
        self.M = np.array([])
        self._d = 0
        self._shape = []
        self._trained = False
        self._inited = False
        self.__prec = 10e-6

    def __str__(self): return 'pPCAMM_k{}_q{}'.format(self.k, self.q)

    def __repr__(self): return 'pPCAMM_k{}_q{}'.format(self.k, self.q)

    def init(self, data_shape: Union[list, tuple, np.ndarray]):
        self._shape = data_shape
        self._d = np.prod(data_shape)
        self.mu = np.zeros((self.k, self._d))
        self.W = np.random.randn(self.k, self._d, self.q)
        self.phi = np.random.rand(self.k) + self.__prec
        self._inited = True

    def __update_M(self):
        self.M = np.linalg.inv(
            self.__transp(self.W) @ self.W + self.phi[:, None] * np.eye(self.q)[None, :, :]
        )

    def __transp(self, mat: np.ndarray) -> np.ndarray:
        return mat.transpose((0, 2, 1))

    def __check_reshape_X(self, X: np.ndarray):
        assert X.shape[1:] == self._shape or X.shape[1] == self._d
        if X.ndim > 2: return X.reshape((X.shape[0], self._d))
        else: return X

    def responsibilities(self, X: np.ndarray, log: bool=False, save_mem: bool=False) -> np.ndarray:
        assert self._inited
        X = self.__check_reshape_X(X)
        res = np.log(self.mix) - 0.5*(self.mahalanobis(X, save_mem=save_mem) + self.logdet())
        res = res - logsumexp(res, axis=0)[None, :]
        return res if log else np.exp(res)

    def mahalanobis(self, X: np.ndarray, save_mem: bool=False) -> np.ndarray:
        """
        Calculates the Mahalanobis distance of the samples X from the cluster k
        :param X: the data to calculate the distance from, as a numpy array with shape [# samples, ...]
        :param save_mem: a flag to indicate whether the computations should be less memory-heavy (making them
                         much less efficient in the process)
        :return: a numpy array with shape [#clusters, # samples] of the distances of each sample from each cluster
        """
        assert self._inited
        X = self.__check_reshape_X(X)
        if not save_mem:
            meaned = X[None, :] - self.mu[:, None, :]
            dist = np.sum(meaned*meaned, axis=2)
            dist2 = meaned @ self.W
            return (dist - np.sum(dist2 * (dist2 @ self.M), axis=2)/self.phi[:, None])/self.phi[:, None]
        else:
            maha = np.zeros((self.k, X.shape[0]))
            for k in range(self.k):
                meaned = X - self.mu[None, k, :]
                dist = np.sum(meaned*meaned, axis=1)
                dist2 = meaned @ self.W[k]
                maha[k, :] = (dist - np.sum(dist2 * (dist2 @ self.M[k]), axis=1)/self.phi[k, None])/self.phi[k, None]
            return maha

    def logdet(self) -> np.ndarray:
        return self._d*np.log(self.phi[:, None, None]) - np.log(self.M)

    def fit(self, X: np.ndarray, iterations: int=10, verbose: bool=True, save_mem: bool=False):
        """
        Fit a GMM to the given datapoints
        :param X: the data to fit to, as a numpy array with shape [# samples, ...]
        :param iterations: the number of iterations to fit the model to the data for
        :param verbose: whether progress should be printed or not
        :return: the trained model
        """
        if not self._inited:
            self.init(X.shape[1:])
        X = self.__check_reshape_X(X)

        if verbose: print('Fitting a pPCAMM model to {} samples with {} clusters, for {} iterations:'
                          .format(X.shape[0], self.k, iterations), flush=True)
        for _ in tqdm.tqdm(range(iterations), disable=not verbose):
            res = self.responsibilities(X, log=True, save_mem=save_mem)
            norm = np.exp(logsumexp(res, axis=1))
            res = np.exp(res)

            self.mix = np.maximum(np.sum(res, axis=1)/X.shape[0], self.__prec)
            self.mix = self.mix / np.sum(self.mix)

            if not save_mem:
                self.mu = np.sum(res[..., None] *
                                 (self.__transp(X[None, :, :]) -
                                  self.M@self.__transp(self.W)@(self.__transp(X[None, :, :])-self.mu[:, :, None])),
                                 axis=2) / norm
                meaned = X[None, :] - self.mu[:, None, :]
                cov = ((res[:, :, None] * (self.__transp(meaned))) @ (meaned @ self.W))/norm[:, None]
                self.W = cov @ np.linalg.inv(self.M @ self.__transp(self.W) @ cov +
                                             np.eye(self.q)[None, :, :]*self.phi[:, None])
                self.phi = (np.sum((res[:, :, None] * meaned @ self.__transp(meaned)) /norm[:, None], axis=1) - \
                            np.sum(self.__transp(self.W) * (cov @ self.M), axis=(1, 2))) / self._d
            else:
                for k in range(self.k):
                    self.mu[k] = np.sum(res[k, :, None] * (X.T - self.M[k] @ self.W[k].T @ (X.T - self.mu[k, :, None])),
                                        axis=1) / norm[k]
                    meaned = X - self.mu[k, None, :]
                    cov = ((res[k, :, None] * meaned.T) @ (meaned @ self.W[k])) / norm[k, None]
                    self.W[k] = cov @ np.linalg.inv(self.M[k] @ self.W[k].T @ cov + np.eye(self.q)*self.phi[k])
                    self.phi[k] = (np.sum(meaned.T @ (res * meaned))/norm[k] - \
                                   np.sum(self.W[k].T * (cov @ self.M[k]))) / self._d
            self.__update_M()
            self.phi = np.maximum(self.phi, self.__prec)

        self._trained = True
        return self

    def generate(self, N: int, return_labels: bool=False, clust: int=None, noise_free: bool=True):
        """
        Generate data from the trained model
        :param N: the number of samples to generate
        :param return_labels: whether the model should return the class labels for the generated data or not
        :param clust: the cluster to generate the samples from. If no cluster is supplied, each generated data point
                      will be sampled according to the mixing probabilities
        :param noise_free: if this is set to False the spherical noise component will be added to the generated samples
        :return: if return_labels is set to False, the return value will be a ndarray with shape [N, ...] of generated
                 samples. If return_labels is set to True, in addition to the first output a ndarray with shape [N,]
                 will be outputted, with the labels that match the generated samples
        """
        assert self._trained, "Model must be trained before trying to generate data from it"
        if clust is None: chosen = np.random.choice(self.k, N, replace=True, p=self.mix)
        else:
            assert clust < self.k
            chosen = np.ones(N)*clust
        labs = [c for c in chosen]
        gen = (self.W[chosen, :, :] @ np.random.randn(N, self.q, 1))[:, :, 0] + self.mu[chosen, :]
        if not noise_free: gen += self.phi[chosen]*np.random.randn(N, self._d)
        if return_labels: return gen.reshape([N] + list(self._shape)), labs
        return gen.reshape([N] + list(self._shape))

    def log_likelihood(self, X: np.ndarray, save_mem: bool=False):
        X = self.__check_reshape_X(X)
        assert self._trained, "Model must be trained before calculating log-likelihood"
        return logsumexp(np.log(self.mix[:, None]) -
                         0.5*(self.mahalanobis(X, save_mem=save_mem) + self.logdet()), axis=0)

    def predict(self, X: np.ndarray, save_mem: bool=False):
        X = self.__check_reshape_X(X)
        assert self._trained, "Model must be trained before predicting labels"
        inds = np.argmax(self.predict_log_proba(X, save_mem=save_mem), axis=0)
        return [i for i in inds]

    def predict_proba(self, X: np.ndarray, save_mem: bool=False):
        X = self.__check_reshape_X(X)
        assert self._trained, "Model must be trained before predicting probabilities"
        return np.exp(self.predict_log_proba(X, save_mem=save_mem))

    def predict_log_proba(self, X: np.ndarray, save_mem: bool=False):
        X = self.__check_reshape_X(X)
        assert self._trained, "Model must be trained before predicting log-probabilities"
        return self.responsibilities(X, log=True, save_mem=save_mem)
