"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import itertools
import sys

import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily, Normal
import torch.nn.functional as F
import math
from scipy.stats import wasserstein_distance

import numpy as np

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features,  self.num_gaussians * (out_features *(out_features + 1))// 2)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        device = self.dummy_param.device
        pi = self.pi(x)
        sigma_tril = torch.zeros([x.shape[0], self.num_gaussians, self.out_features, self.out_features], device=device)
        ti = torch.tril_indices(self.out_features, self.out_features, device=device)
        sigma_tril[:,:,ti[0], ti[1]] = self.sigma(x).view(-1, self.num_gaussians, (self.out_features *(self.out_features + 1))// 2)
        # Ensure diagonal is positive
        sigma_tril[:, :, torch.eye(3).bool()] = torch.exp(torch.diagonal(sigma_tril, dim1=-2, dim2=-1))
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma_tril, mu

    def log_prob(self, x, y):
        pi, sigma, mu = self.forward(x)
        m = MultivariateNormal(mu, scale_tril=sigma)
        mix = Categorical(pi)
        mog = MixtureSameFamily(mix, m)
        return mog.log_prob(y)

    def nll(self, x, y):
        return -self.log_prob(x, y)


def mog_prob(pi, sigma, mu, target):
    m = MultivariateNormal(mu, scale_tril=sigma)
    mix = Categorical(pi)
    mog = MixtureSameFamily(mix, m)
    return torch.exp(mog.log_prob(target))


def mog_mode(pi, sigma, mu):
    from scipy import optimize
    dims = mu.shape[-1]
    mode = np.zeros([dims])
    for d in range(dims):
        mog_d = marginal_mog((pi, sigma, mu), d)

        def nmog_prob_x(x):
            return -mog_prob(*mog_d,
                             torch.Tensor(x)).detach().numpy()

        mode[d] = scipy.optimize.brute(nmog_prob_x, [(-3, 3)])
    return mode


def mog_kl(mog1, mog2):
    x = np.linspace(-3, 3, 120)
    x_batch = torch.Tensor(x.reshape([-1, 1]))

    y1 = mog_prob(*mog1, x_batch).detach().numpy().flatten()
    y2 = mog_prob(*mog2, x_batch).detach().numpy().flatten()
    kl = kl_div(x, y1, y2)
    return kl


def mog_wasserstein(mog1, mog2):
    x = np.linspace(-3, 3, 120)
    x_batch = torch.Tensor(x.reshape([-1, 1]))

    y1 = mog_prob(*mog1, x_batch).detach().numpy().flatten()
    y2 = mog_prob(*mog2, x_batch).detach().numpy().flatten()

    return wasserstein_distance(y1, y2)


def mog_entropy(pi, sigma, mu):
    x = np.linspace(-3, 3, 120)
    x_batch = torch.Tensor(x.reshape([-1, 1]))
    n = len(x)
    y1 = mog_prob(pi, sigma, mu, x_batch).detach().numpy().flatten()
    y2 = np.full([n], 1.0 / 6.0)
    return kl_div(x, y1, y2)


def kl_div(domain, dist1, dist2):
    eps = 0.0000001
    if min(dist1) == 0.0:
        smoothed1 = dist1 + eps
        smoothed1 /= np.trapz(smoothed1, domain)
        kl = kl_div(domain, smoothed1, dist2)
        return kl
    if min(dist2) == 0.0:
        smoothed2 = dist2 + eps
        smoothed2 /= np.trapz(smoothed2, domain)
        kl = kl_div(domain, dist1, smoothed2)
        return kl
    kl = np.trapz(dist1 * np.log2(dist1 / dist2), domain)

    return kl


def batch_mog(mog, n):
    pi, sigma, mu = mog
    return pi.repeat(n, 1), sigma.repeat(n, 1, 1), mu.repeat(n, 1, 1)


def marginal_mog(mog, d):
    pi, sigma, mu = mog
    marginal_mu = mu[:, [d]]
    marginal_sigma = sigma[:, [d]][:, :, [d]]
    return pi, marginal_sigma, marginal_mu


def uniform(width, samples):
    return np.full([samples], 1.0 / width)


def mog_jensen_shanon(mogs):
    # Params are assumed to be batched
    probs = []
    entropies = []
    x = np.linspace(-3, 3, 120)
    x_batch = torch.Tensor(x.reshape([-1, 1]))
    for mog in mogs:
        mog_prob_i = mog_prob(*mog, x_batch).detach().numpy().flatten()
        probs.append(mog_prob_i)
        entropies.append(kl_div(x, mog_prob_i, uniform(6.0, 120)))

    mixture_prob = np.array(probs).sum(0) / len(probs)
    mixture_entropy = kl_div(x, mixture_prob, uniform(6.0, 120))

    return mixture_entropy - (sum(entropies) / len(entropies))


def ens_uncertainty_mode(models, samples):
    out = []

    for model in models:
        out.append(model.forward(samples))
    modes = [[] for _ in range(len(samples))]
    for i, (pi, sigma, mu) in enumerate(out):
        for k in range(len(samples)):
            mode = mog_mode(pi[k], sigma[k], mu[k])
            modes[k].append(mode)
    return np.array(modes).mean(1), np.array(modes).std(1)


def ens_uncertainty_kl(models, samples):
    out = []

    for model in models:
        out.append(model.forward(samples))

    ind = list(range(len(out)))
    dims = out[0][1].shape[-1]
    kl_div = [[[] for _ in range(dims)] for _ in range(len(samples))]
    # Iterate pairs of models
    for i, j in itertools.product(ind, ind):
        if i == j:
            continue
        m0, m1 = out[i], out[j]
        for k in range(len(samples)):
            m0k = m0[0][k], m0[1][k], m0[2][k]
            m1k = m1[0][k], m1[1][k], m1[2][k]
            for d in range(dims):
                kl_div[k][d].append(mog_kl(marginal_mog(m0k, d), marginal_mog(m1k, d)))

    # Mean over all pairs of distances. Out is uncertainty per dimension per sample
    return np.array(kl_div).mean(2)


def ens_uncertainty_w(models, samples):
    out = []

    for model in models:
        out.append(model.forward(samples))

    w_dist = [[[], [], []] for _ in range(len(samples))]
    ind = list(range(len(out)))
    dims = out[0][1].shape[-1]
    # Iterate pairs of models
    for i, j in itertools.product(ind, ind):
        if i == j:
            continue
        m0, m1 = out[i], out[j]
        for k in range(len(samples)):
            m0k = m0[0][k], m0[1][k], m0[2][k]
            m1k = m1[0][k], m1[1][k], m1[2][k]
            for d in range(dims):
                w_dist[k][d].append(mog_wasserstein(marginal_mog(m0k, d), marginal_mog(m1k, d)))

    # Mean over all pairs of distances. Out is uncertainty per dimension per sample
    return np.array(w_dist).mean(2)


def ens_uncertainty_js(models, samples):
    out = []
    for model in models:
        out.append(model.forward(samples))
    dims = out[0][1].shape[-1]
    unc = [[] for _ in range(len(samples))]
    for i in range(len(samples)):
        mogs_per_sample = [(pi[i], sigma[i], mu[i]) for pi, sigma, mu in out]
        for d in range(dims):
            mogs_d = [marginal_mog(mog_d, d) for mog_d in mogs_per_sample]
            unc[i].append(mog_jensen_shanon(mogs_d))

    return np.array(unc)
