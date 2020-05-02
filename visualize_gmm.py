import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.patches import Ellipse
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import argparse
from GMM import GMM

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
parser.add_argument('-s', '--save_path', type=str, default='gmm_vid.mp4',
                    help='where the fitting video will be saved (default: gmm_vid.mp4)')
parser.add_argument('--fps', type=int, default=5,
                    help='frames per second of the video (default: 5')
parser.add_argument('--load_path', type=str, default='',
                    help='load data (as a .npy file) and fit GMM to that data')
parser.add_argument('--print_ll', action='store_false',
                    help='whether to print the average log-likelihood each iteration (default: False)')
parser.add_argument('--show_clusters', action='store_false',
                    help='whether to plot the Gaussians or not (default: True)')

colors = plt.get_cmap('tab20').colors


def create_frame():
    gca = plt.gca()
    res = np.array(gmm.predict(X))
    for i in range(k):
        inds = res == i
        if np.any(inds):
            plt.scatter(X[inds, 0], X[inds, 1], 10, c=colors[i], alpha=.5, marker='.')
            if args.show_clusters:
                plt.plot(gmm.mu[i, 0], gmm.mu[i, 1], marker='+', markersize=5, color=colors[i])
                v, w = np.linalg.eigh(gmm.cov[i])
                ang = 180. * np.arctan2(w[0, 1], w[0, 0]) / np.pi
                v = 2*np.sqrt(2)*np.sqrt(v)
                e = Ellipse(gmm.mu[i], v[0], v[1], 180 + ang, lw=2, facecolor='None',
                            edgecolor=colors[i])
                gca.add_patch(e)
    plt.xlim(xlims)
    plt.ylim(ylims)


# parse script arguments
args = parser.parse_args()
k = args.num_clusters
its = args.iterations

# load/create data to fit to
if args.load_path != '':
    X = np.load(args.load_path)
else:
    theta = np.random.rand(args.means)*2*np.pi
    r = (.85*np.random.rand(args.means) + .15)*args.max_radius
    mu = np.zeros((args.means, 2))
    mu[:, 0] = r*np.cos(theta)
    mu[:, 1] = r*np.sin(theta)

    covs = np.random.randn(args.means, 2, 2)
    covs = covs @ covs.transpose((0, 2, 1))
    covs += 0.1*np.eye(2)[None, ...]

    split = np.random.rand(args.means)
    split = np.ceil(args.N*split/np.sum(split))
    split = [0] + [min(int(np.sum(split[:i+1])), args.N) for i in range(len(split))]

    X = np.zeros((args.N, 2))
    for i in range(args.means):
        X[split[i]:split[i+1]] = np.random.multivariate_normal(mean=mu[i], cov=covs[i], size=split[i+1]-split[i])

# define axes limits
mid = np.mean(X, axis=0)
dist = 1.25*np.max(X-mid, axis=0)
xlims = [mid[0]-dist[0], mid[0]+dist[0]]
ylims = [mid[1]-dist[1], mid[1]+dist[1]]

# initialize the model
gmm = GMM(k)

# create the writer
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='GMM fitting visualization', artist='')
writer = FFMpegWriter(fps=args.fps, metadata=metadata)

# write frames
fig = plt.figure()
with writer.saving(fig, args.save_path, 100):
    for i in range(its):
        gmm = gmm.fit(X, iterations=1, verbose=False)
        if args.print_ll: print("The average log-likelihood for iteration {}/{} was {:.3f}"
                                .format(i+1, its, np.mean(gmm.log_likelihood(X))), flush=True)
        plt.clf()
        create_frame()
        writer.grab_frame()
