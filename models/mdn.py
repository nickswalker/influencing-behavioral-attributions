"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import itertools

import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
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
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, x):
        pi = self.pi(x)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def mog_prob(pi, sigma, mu, target):
    prob = pi * gaussian_probability(sigma, mu, target)
    return torch.sum(prob, dim=1)


def sample_mog_n(n, pi, sigma, mu):
    all = [sample_mog(pi, sigma, mu) for _ in range(n)]
    all = [torch.unsqueeze(x, 1) for x in all]
    return torch.cat(all, 1)


def mog_mean(pi, mu):
    return torch.sum(pi @ mu, 1)


def mog_mode(pi, sigma, mu):
    from scipy import optimize
    import numpy as np
    mode = np.zeros([pi.shape[0]])
    for i in range(pi.shape[0]):
        def nmog_prob_x(x):
            return -mog_prob(pi[i, :].unsqueeze(0), sigma[i].unsqueeze(0), mu[i].unsqueeze(0),
                             torch.Tensor(x)).detach().numpy()

        mode[i] = scipy.optimize.brute(nmog_prob_x, [(-3, 3)])
    return mode


def mog_kl(mog1, mog2):
    x = np.linspace(-3, 3, 120)
    x_batch = torch.Tensor(x.reshape([-1, 1]))
    n = len(x)
    mog1 = mog1[0].repeat(n, 1), mog1[1].repeat(n, 1, 1), mog1[2].repeat(n, 1, 1)
    mog2 = mog2[0].repeat(n, 1), mog2[1].repeat(n, 1, 1), mog2[2].repeat(n, 1, 1)

    y1 = mog_prob(*mog1, x_batch).detach().numpy().flatten()
    y2 = mog_prob(*mog2, x_batch).detach().numpy().flatten()
    kl = kl_div(x, y1, y2)
    kl_2 = kl_div(x, y2, y1)
    return kl + kl_2 / 2.0, kl, kl_2

def mog_wasserstein(mog1, mog2):
    x = np.linspace(-3, 3, 120)
    x_batch = torch.Tensor(x.reshape([-1, 1]))
    n = len(x)
    mog1 = mog1[0].repeat(n, 1), mog1[1].repeat(n, 1, 1), mog1[2].repeat(n, 1, 1)
    mog2 = mog2[0].repeat(n, 1), mog2[1].repeat(n, 1, 1), mog2[2].repeat(n, 1, 1)

    y1 = mog_prob(*mog1, x_batch).detach().numpy().flatten()
    y2 = mog_prob(*mog2, x_batch).detach().numpy().flatten()

    return wasserstein_distance(y1, y2)


def mog_entropy(pi, sigma, mu):
    x = np.linspace(-3, 3, 120)
    x_batch = torch.Tensor(x.reshape([-1, 1]))
    n = len(x)
    mog1 = pi.repeat(n, 1), sigma.repeat(n, 1, 1), mu.repeat(n, 1, 1)

    y1 = mog_prob(*mog1, x_batch).detach().numpy().flatten()
    y2 = np.full([n], 1.0 / 6.0)
    return kl_div(x, y1, y2)


def kl_div(domain, dist1, dist2):
    kl = np.trapz(dist1 * np.log2(dist1 / dist2), domain)
    return kl


def mog_jensen_shanon(mogs):
    # Params are assumed to be batched
    probs = []
    entropies = []
    x = np.linspace(-3, 3, 120)
    x_batch = torch.Tensor(x.reshape([-1, 1]))
    n = len(x)
    for pi, sigma, mu in mogs:
        mog = pi.repeat(n, 1), sigma.repeat(n, 1, 1), mu.repeat(n, 1, 1)
        mog_prob_i = mog_prob(*mog, x_batch).detach().numpy().flatten()
        probs.append(mog_prob_i)
        entropies.append(kl_div(x, mog_prob_i, np.full([1, n], 1.0 / 6.0)))

    mixture_prob = np.array(probs).sum(0) / len(probs)
    mixture_entropy = kl_div(x, mixture_prob, np.full([n], 1.0 / 6.0))

    return mixture_entropy - (sum(entropies) / len(entropies))


def sample_mog(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i, idx]).add(mu[i, idx])
    return sample


def ens_uncertainty_mode(models, samples):
    out = []

    for model in models:
        out.append(model.forward(samples))
    modes = [[] for _ in range(len(samples))]
    means = [[] for _ in range(len(samples))]
    for i, (pi, sigma, mu) in enumerate(out):
        for k in range(len(samples)):
            mode = mog_mode(pi[k], sigma[k], mu[k])
            mean = mog_mean(pi[k], mu[k]).squeeze()
            modes[k].append(mode)
            means[k].append(mean.detach().numpy())
    return np.array(modes).mean(1), np.array(modes).std(1), np.array(means).mean(1), np.array(means).std(1)


def ens_uncertainty_kl(models, samples):
    out = []

    for model in models:
        out.append(model.forward(samples))

    kl_div = [[[], [], []] for _ in range(len(samples))]
    ind = list(range(len(out)))
    dims = out[0][0].shape[-1]
    # Iterate pairs of models
    for i, j in itertools.product(ind, ind):
        if i == j:
            continue
        m0, m1 = out[i], out[j]
        for k in range(len(samples)):
            for d in range(dims):
                m0k = m0[0][k][d], m0[1][k][d], m0[2][k][d]
                m1k = m1[0][k][d], m1[1][k][d], m1[2][k][d]
                kl_div[k][d].append(mog_kl(m0k, m1k)[0])

    # Mean over all pairs of distances. Out is uncertainty per dimension per sample
    return np.array(kl_div).mean(2)


def ens_uncertainty_w(models, samples):
    out = []

    for model in models:
        out.append(model.forward(samples))

    w_dist = [[[], [], []] for _ in range(len(samples))]
    ind = list(range(len(out)))
    dims = out[0][0].shape[-1]
    # Iterate pairs of models
    for i, j in itertools.product(ind, ind):
        if i == j:
            continue
        m0, m1 = out[i], out[j]
        for k in range(len(samples)):
            for d in range(dims):
                m0k = m0[0][k][d], m0[1][k][d], m0[2][k][d]
                m1k = m1[0][k][d], m1[1][k][d], m1[2][k][d]
                w_dist[k][d].append(mog_wasserstein(m0k, m1k))

    # Mean over all pairs of distances. Out is uncertainty per dimension per sample
    return np.array(w_dist).mean(2)


def ens_uncertainty_js(models, samples):
    out = []
    for model in models:
        out.append(model.forward(samples))
    dims = out[0][0].shape[-1]
    unc = [[] for _ in range(len(samples))]
    for i in range(len(samples)):
        for d in range(dims):
            mogs_per_sample = [(model_out[0][i][d], model_out[1][i][d], model_out[2][i][d]) for model_out in out]
            unc[i].append(mog_jensen_shanon(mogs_per_sample))

    return np.array(unc)
