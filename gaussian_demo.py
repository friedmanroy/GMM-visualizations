import numpy as np
from mosaic import mosaic
from matplotlib import pyplot as plt
from models.classGMM import pPCA_cGMM as classGMM


def gen_labeled(N, n_shapes: int=5, im_sz: int=20, sqr_size: int=7, bg_mean: float=.2, fg_mean: float=.8,
                 bg_std: np.array=np.array([.2, .05, .1])):
    def _shape_creator(mode: int):
        if mode == 0:
            ret_val = np.ones((sqr_size, sqr_size))
        elif mode==1:
            ret_val = np.ones((sqr_size, sqr_size))
            ret_val[1:-1, 1:-1] = 0
        elif mode==2:
            ret_val = np.zeros((sqr_size, sqr_size))
            ret_val[0, :] = 1
            ret_val[sqr_size // 2, :] = 1
            ret_val[:sqr_size // 2, 0] = 1
        elif mode==3:
            ret_val = np.zeros((sqr_size, sqr_size))
            ret_val[sqr_size // 4:3 * sqr_size // 4, :2 * sqr_size // 3] = 1
        else:
            mode -= 4
            even = sqr_size % 2 == 0
            X, Y = np.meshgrid(np.arange(-(sqr_size // 2), sqr_size // 2 + 1 if not even else sqr_size // 2),
                               np.arange(-(sqr_size // 2), sqr_size // 2 + 1 if not even else sqr_size // 2))
            if mode < sqr_size:
                mask = np.clip(np.abs(X) + np.abs(Y) - 1, 0, sqr_size - 1)
            elif mode <= sqr_size < 2*sqr_size:
                mask = np.abs(X * Y + X - Y) // 2 + sqr_size
            else:
                mask = mode * np.random.randint(0, 2, sqr_size ** 2).reshape((sqr_size, sqr_size))
            ret_val = np.zeros((sqr_size, sqr_size))
            ret_val[mask == mode] = 1
        return ret_val
    mix = np.maximum(np.random.rand(n_shapes), .2)
    mix /= np.sum(mix)
    print('Generating data with mix:', mix)
    labels = np.random.choice(n_shapes, N, p=mix)
    pos = np.ceil((np.random.rand(n_shapes, 2)-.5)*(im_sz//2-sqr_size//2))
    pos = pos.astype(int)
    gen = []
    for l in labels:
        im = bg_mean * np.ones((im_sz, im_sz, 3))
        im += (bg_std * np.random.randn(3))[None, None, :]
        sh = _shape_creator(l)
        tmp = np.where(sh == 1)
        im[im_sz//2-sqr_size//2-pos[l,0]+ tmp[0], im_sz//2-sqr_size//2-pos[l,1]+tmp[1]] = fg_mean
        gen.append(np.clip(im, 0, 1))
    return np.array(gen), labels


def classifier_demo(N: int, n_classes: int, train_ratio: float=.8, latent_dim: int=50):
    ims, labs = gen_labeled(N, n_shapes=n_classes)
    train = slice(0, int(np.ceil(N*train_ratio)))
    test = slice(int(np.ceil(N*train_ratio)), N)

    gmm = classGMM(latent_dim=latent_dim).fit(ims[train], labs[train])
    print('cGMM trained with mix:', gmm.mix)
    print('Avg. train accuracy:', gmm.score(ims[train], labs[train]))
    print('Avg. test accuracy:', gmm.score(ims[test], labs[test]))

    plt.figure()
    plt.subplot(121)
    plt.imshow(mosaic(ims[:49]))
    plt.axis('off')
    plt.title('original data')
    plt.subplot(122)
    plt.imshow(mosaic(gmm.generate(49)))
    plt.axis('off')
    plt.title('generated data')
    plt.tight_layout()

    preds = gmm.predict(ims[test])
    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(ims[test][i])
        plt.axis('off')
        plt.title('lab: {}, pred: {}'.format(labs[test][i], preds[i]))
    plt.tight_layout()

    plt.figure()
    for i in range(n_classes):
        cls = mosaic(ims[labs==i][:10], cols=1)
        plt.subplot(1, 2*n_classes, 2*(i+1)-1)
        plt.imshow(cls)
        plt.axis('off')
        plt.title('real')

        cls = mosaic(gmm.generate(10, label=i), cols=1)
        plt.subplot(1, 2*n_classes, 2*(i + 1))
        plt.imshow(cls)
        plt.axis('off')
        plt.title('learned')
    plt.tight_layout()

    plt.show()


def squares_demo(N: int, bg_mean: float=.2, fg_mean: float=.8,
                 bg_std: np.array=np.array([.2, .5, .3]),
                 fg_std: np.array = np.array([.3, .1, .6])):
    squares = bg_mean * np.ones((N, 20, 20, 3))
    for i in range(N):
        squares[i] += (np.random.randn(3)*bg_std)[None, None, :]
        squares[i, 7:13, 7:13] = fg_mean
        squares[i, 7:13, 7:13] += (np.random.randn(3)*fg_std)[None, None, :]

    orig_examp = mosaic(squares[:36])

    squares = squares.reshape(N, -1)
    mu = np.mean(squares, axis=0)
    cov = (squares - mu).T @ (squares - mu) / N

    gen_examp = mosaic(np.random.multivariate_normal(mu, cov, 36).reshape((36, 20, 20, 3)))

    plt.figure()

    plt.subplot(131)
    plt.imshow(orig_examp)
    plt.axis('off')
    plt.title('original samples')

    plt.subplot(132)
    plt.imshow(gen_examp)
    plt.axis('off')
    plt.title('generated samples')

    plt.subplot(133)
    plt.imshow(mu.reshape((20, 20, 3)))
    plt.axis('off')
    plt.title('learned mean')

    plt.show()


def demo_MNIST(latent: int=50):
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape, X_test.shape)
    print(np.unique(y_train), np.unique(y_test))

    gmm = classGMM(latent_dim=latent).fit(X_train, y_train)
    print('cGMM trained with mix:', gmm.mix)
    print('Avg. train accuracy:', gmm.score(X_train, y_train))
    print('Avg. test accuracy:', gmm.score(X_test, y_test))

    plt.figure(dpi=300)
    plt.subplot(121)
    plt.imshow(mosaic(X_train[:49]), cmap='gray')
    plt.axis('off')
    plt.title('original data')
    plt.subplot(122)
    plt.imshow(mosaic(gmm.generate(49)), cmap='gray')
    plt.axis('off')
    plt.title('generated data')
    plt.tight_layout()

    preds = gmm.predict(X_test)
    plt.figure(dpi=300)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_test[i], cmap='gray')
        plt.axis('off')
        plt.title('lab: {}, pred: {}'.format(y_test[i], preds[i]))
    plt.tight_layout()

    plt.figure(dpi=300)
    for i in range(10):
        cls = mosaic(X_train[y_train == i][:10], cols=1)
        plt.subplot(1, 2 * 10, 2 * (i + 1) - 1)
        plt.imshow(cls, cmap='gray')
        plt.axis('off')

        cls = mosaic(gmm.generate(10, label=i), cols=1)
        plt.subplot(1, 2 * 10, 2 * (i + 1))
        plt.imshow(cls, cmap='gray')
        plt.axis('off')
    plt.tight_layout()

    plt.show()


# simple MVN demo
squares_demo(1000)

# low rank MVN demo
squares_demo(1000, fg_std=np.array([0, 0, 0]))

# demo for class-GMM
classifier_demo(60000, 3, .9, 5)

# demo using MNIST
demo_MNIST(10)
