import numpy as np


class SynGen:
    def __init__(self, im_sz: int=20, sqr_sz: int=10, n_modes: int=1, n_pos: int=1, n_latent: int=5,
                 eps: float=.1, latent_weight: float=.3, bg_color: list=None, fg_color: list=None):

        self.im_sz = im_sz
        self.sqr_sz = sqr_sz
        self.n_modes = n_modes
        self.n_pos = n_pos
        self.n_latent = n_latent
        self.eps = eps
        self.latent_weight = latent_weight
        self.fg_color = fg_color
        self.bg_color = bg_color

        self._inits()

        self.sz = (im_sz, im_sz, 3)
        if n_latent > 0: self.covs = self._create_covs()
        else: self.covs = []
        self._modes = [self._shape_creator(i, sqr_sz)[:, :, None] * self._fg[i][None, None, :] for i in range(n_modes)]

        self.mu = [self._place(i, 0) for i in range(n_modes)]

    def __call__(self, N: int) -> list:
        return self.generate(N)

    def _create_covs(self):
        covs = []
        for i in range(self.n_modes):
            cov = (self.latent_weight * np.ones((self.im_sz**2, 3, self.n_latent))
                   * np.random.randn(3, self.n_latent)[None, :, :]).reshape((np.prod(self.sz), self.n_latent))
            covs.append(cov)
        return covs

    def _place(self, mode, pos):
        im = np.zeros(self.sz) * self._bg[mode][None, None, :]
        im[self._x_pos[pos]:self._x_pos[pos] + self.sqr_sz, self._y_pos[pos]:self._y_pos[pos] + self.sqr_sz, :] = \
            self._modes[mode]
        return im

    @staticmethod
    def _shape_creator(mode: int, size: int):
        if mode == 0:
            ret_val = np.ones((size, size))
        elif mode==1:
            ret_val = np.ones((size, size))
            ret_val[1:-1, 1:-1] = 0
        elif mode==2:
            ret_val = np.zeros((size, size))
            ret_val[0, :] = 1
            ret_val[size//2, :] = 1
            ret_val[:size//2, 0] = 1
        elif mode==3:
            ret_val = np.zeros((size, size))
            ret_val[size//4:3*size//4, :2*size//3] = 1
        else:
            mode -= 4
            even = size % 2 == 0
            X, Y = np.meshgrid(np.arange(-(size//2), size//2 + 1 if not even else size//2),
                               np.arange(-(size//2), size//2 + 1 if not even else size//2))
            if mode < size:
                mask = np.clip(np.abs(X) + np.abs(Y) - 1, 0, size-1)
            elif mode <= size < 2*size:
                mask = np.abs(X * Y + X - Y) // 2 + size
            else:
                mask = mode * np.random.randint(0, 2, size ** 2).reshape((size, size))
            ret_val = np.zeros((size, size))
            ret_val[mask == mode] = 1
        return ret_val

    def generate(self, N: int) -> list:
        data = []
        s = np.random.choice(self.n_modes, N, replace=True)
        positions = np.random.choice(self.n_pos, N, replace=True)
        for i in range(N):
            im = self._place(s[i], positions[i])
            if self.n_latent > 0:
                im += (self.covs[s[i]] @ np.random.randn(self.n_latent)).reshape(self.sz)
            im += self.eps * np.random.randn(np.prod(self.sz)).reshape(self.sz)
            data.append(im)
        return data

    def _inits(self):
        assert self.im_sz > self.sqr_sz, "The image size ({}) must be larger than the shape's " \
                                         "size ({}).".format(self.im_sz, self.sqr_sz)
        assert (np.array(self.bg_color).shape[0] == 3) if self.bg_color is not None \
            else True, "The given background color must" \
                       " be a 3-tuple of (r,g,b); instead " \
                       "{} was given.".format(self.bg_color)
        assert (np.array(self.fg_color).shape[0] == 3) if self.fg_color is not None \
            else True, "The given foreground color must" \
                       " be a 3-tuple of (r,g,b); instead " \
                       "{} was given.".format(self.fg_color)
        if self.n_latent < 0: self.n_latent = 0
        if self.n_modes < 1: self.n_modes = 1
        if self.n_pos < 1: self.n_pos = 1

        if self.bg_color is None:
            self._bg = [.5*np.random.rand(3) for _ in range(self.n_modes)]
        else: self._bg = [np.array(self.bg_color) for _ in range(self.n_modes)]
        if self.fg_color is None:
            self._fg = [.4*np.random.rand(3) + .6 for _ in range(self.n_modes)]
        else: self._fg = [np.array(self.fg_color) for _ in range(self.n_modes)]

        self._x_pos = np.random.choice(self.im_sz-self.sqr_sz, max(self.n_pos, self.im_sz-self.sqr_sz), replace=False)
        self._y_pos = np.random.choice(self.im_sz-self.sqr_sz, max(self.n_pos, self.im_sz-self.sqr_sz), replace=False)

    def get_fullcov(self, add_noise: bool=True):
        return [c@c.T + (np.eye(c.shape[0])*self.eps if add_noise else 0) for c in self.covs]


def shapes_generator(im_sz, square_sz, N, n_pos=1, n_shape=5, eps=0.1) -> list:
    return SynGen(im_sz, square_sz, n_pos=n_pos, n_modes=n_shape, eps=eps, n_latent=0,
                  bg_color=[0, 0, 0], fg_color=[.8, .8, .8])(N)


def noisy_squares_generator(im_sz, square_sz, N, n_pos=1, eps=0.1) -> list:
    return SynGen(im_sz, square_sz, n_pos=n_pos, n_modes=1, eps=eps, n_latent=0,
                  bg_color=[0, 0, 0], fg_color=[.8, .8, .8])(N)
