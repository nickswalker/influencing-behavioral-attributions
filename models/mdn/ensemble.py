import itertools

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from models.mdn.lit_mdn import LitMDN
from models.mdn.mdn import mog_desc, marginal_mog_log_prob, mog_wasserstein, marginal_mog, mog_jensen_shanon, mog_log_prob


class MDNEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        out = []
        for m in self.models:
            out.append(m.forward(x))
        pi = torch.stack([p[0] for p in out], dim=1)
        sigma = torch.stack([p[1] for p in out], dim=1)
        mu = torch.stack([p[2] for p in out], dim=1)
        return pi, sigma, mu

    def freeze(self):
        for m in self.models:
            for param in m.parameters():
                    param.requires_grad_(False)

    def log_prob(self, x, y):
        out = self.forward(x)
        prob = mog_log_prob(*out,y)
        return prob

    def prob(self, x, y):
        return torch.exp(self.log_prob(x, y))

    def mean_prob(self, x, y):
        return torch.mean(torch.exp(self.log_prob(x, y)),dim=1)

    def mean_nll(self, x, y):
        ens_probs = self.mean_prob(x, y)
        return -torch.log(ens_probs).mean(0)

    @staticmethod
    def load_ensemble(path):
        ens = []
        for i in range(8):
            ens.append(LitMDN.load_from_checkpoint(path + "/mdn_" + str(i) + ".ckpt"))
        return MDNEnsemble(ens)


def ens_uncertainty_mean(ens, samples):
    out = ens.forward(samples)
    means = mog_desc(*out)[0]
    return means.std(1)


def ens_uncertainty_mode(ens, samples):
    out = ens.forward(samples)
    x = torch.linspace(-3, 3, 600)
    probs = marginal_mog_log_prob(*out, x.reshape([-1, 1, 1]))
    modes = x[torch.argmax(probs, dim=-1)]
    # Variance of ens modes per dim
    return modes.std(1)


def ens_uncertainty_kl(ens, samples):
    n = len(ens.models)
    probs = marginal_mog_log_prob(*ens.forward(samples), torch.linspace(-3, 3, 120).reshape([-1, 1, 1]))
    pairwise_kl = F.kl_div(probs.unsqueeze(1), probs.unsqueeze(2), reduction='none', log_target=True)
    pairwise_kl = torch.trapz(pairwise_kl, dx=0.05)
    offdiag_i = torch.hstack([torch.tril_indices(n, n, -1), torch.triu_indices(n, n, 1)])
    offdiag_kl = pairwise_kl[:, offdiag_i[0], offdiag_i[1]]
    # Mean over all pairs of distances. Out is uncertainty per dimension per sample
    return offdiag_kl.mean(1)


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


