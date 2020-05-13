import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import argparse
from models.classGMM import cGMM
matplotlib_axes_logger.setLevel('ERROR')


parser = argparse.ArgumentParser(description='Visualizes the performance of class-GMM on simple data')

parser.add_argument('-k', '--num_classes', type=int, default=2,
                    help='the number of classes the GMM will be trained on(default: 2)')
parser.add_argument('-N', type=int, default=1000,
                    help='number of data points to generate (default: 1000')
parser.add_argument('-r', '--train_frac', type=float, default=.8,
                    help='the fraction of points that should be used to train (default: .8')
parser.add_argument('--max_radius', type=float, default=10,
                    help='max radius that the generated points will lie in (default: 10')
parser.add_argument('--hide_covs', action='store_true',
                    help='whether to hide the fitted Gaussians covariances or not (default: False)')
parser.add_argument('--hide_means', action='store_true',
                    help='whether to hide the fitted Gaussians means or not (default: False)')

colors = plt.get_cmap('Set1').colors


def get_ellipse(mean, cov, color, linestyle='-'):
    v, w = np.linalg.eigh(cov)
    ang = 180. * np.arctan2(w[0, 1], w[0, 0]) / np.pi
    v = 2 * np.sqrt(2) * np.sqrt(v)
    return Ellipse(mean, v[0], v[1], 180 + ang, lw=2, facecolor='None',
                   edgecolor=color, linestyle=linestyle)


# parse script arguments
args = parser.parse_args()
assert 0 < args.train_frac <= 1, 'The fraction of points that will be used to train must be bigger than 0 ' \
                                 'and smaller or equal to 1. Instead, {} was given.'.format(args.train_frac)
if args.max_radius < 0:
    import sys
    print('The mininmal possible radius is 0. Instead, {} was given, which will be clipped to 0.'
          .format(args.max_radius), file=sys.stderr)
    args.max_radius = 0

k = args.num_classes

# create data to fit to
theta = np.random.rand(k)*2*np.pi
r = (.85*np.random.rand(k) + .15)*args.max_radius
mu = np.zeros((k, 2))
mu[:, 0] = r*np.cos(theta)
mu[:, 1] = r*np.sin(theta)

covs = np.random.randn(k, 2, 2)
covs = covs @ covs.transpose((0, 2, 1))
covs += 0.1*np.eye(2)[None, ...]

rmu, rcov = mu, covs
split = np.random.rand(k)
split = np.ceil(args.N*split/np.sum(split))
split = [0] + [min(int(np.sum(split[:i+1])), args.N) for i in range(len(split))]

X = np.zeros((args.N, 2))
y = np.zeros(args.N)
for i in range(k):
    X[split[i]:split[i+1]] = np.random.multivariate_normal(mean=mu[i], cov=covs[i], size=split[i+1]-split[i])
    y[split[i]:split[i+1]] = i


# define axes limits
mid = np.mean(X, axis=0)
dist = 1.25*np.sqrt(np.max((X-mid)**2, axis=0))
xlims = [mid[0]-dist[0], mid[0]+dist[0]]
ylims = [mid[1]-dist[1], mid[1]+dist[1]]

train = np.ones(args.N).astype(bool)
train[np.random.choice(args.N, int(np.floor(1-args.train_frac*args.N)), replace=False)] = False
test = ~train

gmm = cGMM().fit(X[train], y[train], verbose=False)
preds = np.array(gmm.predict(X)).astype(int)
tr_score, te_score = gmm.score(X[train], y[train]), gmm.score(X[test], y[test])

plt.figure(dpi=100)
gca = plt.gca()
for i in range(k):
    p = preds == i
    gt = y == i
    if np.any(p):
        plt.scatter(X[p & gt, 0], X[p & gt, 1], 15, c=colors[i % len(colors)], alpha=.5, marker='o')
        plt.scatter(X[p & ~gt, 0], X[p & ~gt, 1], 15, c=colors[i % len(colors)], alpha=.5, marker='x')
    if not args.hide_means:
        plt.plot(gmm.mu[i, 0], gmm.mu[i, 1], marker='+', markersize=15, color=colors[i % len(colors)])
    if not args.hide_covs:
        gca.add_patch(get_ellipse(gmm.mu[i], gmm.cov[i], colors[i % len(colors)]))
plt.xlim(xlims)
plt.ylim(ylims)
plt.axis('off')
plt.text(xlims[0], ylims[1], "train acc.={:.2f}, test acc.={:.2f}".format(tr_score, te_score),
         horizontalalignment='left', verticalalignment='top')
plt.show()
