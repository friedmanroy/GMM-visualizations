import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as manimation
from matplotlib.patches import Ellipse
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import argparse
from models.GMM import GMM
matplotlib_axes_logger.setLevel('ERROR')

parser = argparse.ArgumentParser(description='Visualizes how a GMM converges and how it is trained')

parser.add_argument('-k', '--num_clusters', type=int, default=5,
                    help='the number of clusters in the GMM that will be trained (default: 2)')
parser.add_argument('-m', '--means', type=int, default=5,
                    help='number of means to use to generate data (default: 2')
parser.add_argument('-i', '--iterations', type=int, default=50,
                    help='number of iterations to fit the model (default: 20')
parser.add_argument('-N', type=int, default=1000,
                    help='number of data points to generate (default: 1000')
parser.add_argument('--max_radius', type=float, default=10,
                    help='max radius that the means will lie in (default: 10')
parser.add_argument('-s', '--save_path', type=str, default='gmm_vid.gif',
                    help='where the fitting video will be saved (default: gmm_vid.gif)')
parser.add_argument('--fps', type=int, default=10,
                    help='frames per second of the video (default: 5')
parser.add_argument('--load_path', type=str, default='',
                    help='load data (as a .npy file) and fit GMM to that data')
parser.add_argument('--print_ll', action='store_true',
                    help='whether to print the average log-likelihood each iteration (default: False)')
parser.add_argument('--hide_covs', action='store_true',
                    help='whether to hide the fitted Gaussians covariances or not (default: False)')
parser.add_argument('--hide_means', action='store_true',
                    help='whether to hide the fitted Gaussians means or not (default: False)')
parser.add_argument('--hide_real', action='store_true',
                    help='whether to hide the true Gaussian placements or not (default: False)')

colors = plt.get_cmap('Set1').colors


def get_ellipse(mean, cov, color, linestyle='-'):
    v, w = np.linalg.eigh(cov)
    ang = 180. * np.arctan2(w[0, 1], w[0, 0]) / np.pi
    v = 2 * np.sqrt(2) * np.sqrt(v)
    return Ellipse(mean, v[0], v[1], 180 + ang, lw=2, facecolor='None',
                   edgecolor=color, linestyle=linestyle)


def create_frame(it: int):
    gca = plt.gca()
    plt.text(xlims[0], ylims[1], "iter {}/{}".format(it+1, its), horizontalalignment='left',
             verticalalignment='top')
    res = np.array(gmm.predict(X))

    if len(rmu) > 0 and not args.hide_real:
        for i in range(len(rmu)):
            gca.add_patch(get_ellipse(rmu[i], rcov[i], [0.7, 0.7, 0.7], '--'))

    for clust in range(k):
        inds = res == clust
        if np.any(inds):
            plt.scatter(X[inds, 0], X[inds, 1], 10, c=colors[clust%len(colors)], alpha=.5, marker='.')
        if not args.hide_means:
            plt.plot(gmm.mu[clust, 0], gmm.mu[clust, 1], marker='+', markersize=5, color=colors[clust%len(colors)])
        if not args.hide_covs:
            gca.add_patch(get_ellipse(gmm.mu[clust], gmm.cov[clust], colors[clust%len(colors)]))

    plt.xlim(xlims)
    plt.ylim(ylims)


# parse script arguments
args = parser.parse_args()
k = args.num_clusters
its = args.iterations

# load/create data to fit to
rmu = []
rcov = []
if args.load_path != '': X = np.load(args.load_path)
else:
    theta = np.random.rand(args.means)*2*np.pi
    r = (.85*np.random.rand(args.means) + .15)*args.max_radius
    mu = np.zeros((args.means, 2))
    mu[:, 0] = r*np.cos(theta)
    mu[:, 1] = r*np.sin(theta)

    covs = np.random.randn(args.means, 2, 2)
    covs = covs @ covs.transpose((0, 2, 1))
    covs += 0.1*np.eye(2)[None, ...]

    rmu, rcov = mu, covs
    split = np.random.rand(args.means)
    split = np.ceil(args.N*split/np.sum(split))
    split = [0] + [min(int(np.sum(split[:i+1])), args.N) for i in range(len(split))]

    X = np.zeros((args.N, 2))
    for i in range(args.means):
        X[split[i]:split[i+1]] = np.random.multivariate_normal(mean=mu[i], cov=covs[i], size=split[i+1]-split[i])


# define axes limits
mid = np.mean(X, axis=0)
dist = 1.25*np.sqrt(np.max((X-mid)**2, axis=0))
xlims = [mid[0]-dist[0], mid[0]+dist[0]]
ylims = [mid[1]-dist[1], mid[1]+dist[1]]

# initialize the model
gmm = GMM(k)

# create the writer
FFMpegWriter = manimation.writers['imagemagick']
metadata = dict(title='GMM fitting visualization', artist='')
writer = FFMpegWriter(fps=args.fps, metadata=metadata)

# write frames
fig = plt.figure()
with writer.saving(fig, args.save_path, 100):
    pbar = tqdm(range(its))
    for i in pbar:
        gmm = gmm.fit(X, iterations=1, verbose=False)
        if args.print_ll: pbar.set_postfix_str("log-likelihood: {:.3f}".format(np.mean(gmm.log_likelihood(X))))
        plt.clf()
        create_frame(i)
        writer.grab_frame()
