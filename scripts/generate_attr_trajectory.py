import glob
import itertools
import json
import os
import random
import sys
from functools import partial

import scipy
import torch
import pandas as pd

import numpy as np
from scipy.spatial.distance import cdist

from models.mdn.mdn import normal, marginal_mog_log_prob
from models.mdn.lit_mdn import LitMDN
from models.mdn.ensemble import MDNEnsemble, ens_uncertainty_kl
from models.util import gen_samples
from processing.loading import load_trajectory_pool, load_demo_pool, process_turk_files
from processing.mappings import attributions
from search.coverage import trajectory_features, trajectory_cost, CoverageNode
from search.metric import coverage_manhattan
from search.routine import hill_descend, batch_hill_descend, astar
from search.sampling import sample_neighbors, sample_neighbors_for_pool
from search.trajectory import TrajectoryNode, ANY_PLAN
from search.util import load_map
import torch.nn.functional as F

from matplotlib import pyplot as plt

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)


class MyBounds(object):
    def __init__(self, xmax=np.full([11], 1), xmin=np.full([11], 0)):
        self.xmax = xmax
        self.xmin = xmin

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def batch_min_div(nodes, goal_dist, x, d):
    node_feats = np.array([node.features for node in nodes], dtype=np.float)
    node_batch = torch.Tensor(node_feats)
    # dists = ens.prob(node_batch, x)
    # dists = dists.T.reshape(-1, 120,120,120)
    # pred = ens.mean_prob(node_batch, x)
    p_marg = marginal_mog_log_prob(*ens.forward(node_batch), x.reshape([-1, 1, 1]))
    # TODO: FIX to mean after exp
    pred = p_marg.mean(1)
    pred_d = pred[:, d]

    kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
    kl = torch.trapz(kl, x)
    print(kl.min())
    unc = ens_uncertainty_kl(ens, node_batch)
    # We want distribution to match, so drive kl to 0
    return kl.detach().numpy()


def optimize_feats(model, goal_dist, d, x):
    inputs = torch.rand((10000,11), requires_grad=True)

    p_marg = marginal_mog_log_prob(*ens.forward(torch.clamp(inputs, 0, 1)), x.reshape([-1, 1, 1]))
    # TODO: FIX to mean after exp
    pred = p_marg.mean(1)
    pred_d = pred[:, d]

    kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
    kl = torch.trapz(kl, x)
    inputs = inputs[torch.argmin(kl, 0)]
    inputs = inputs.reshape(1, -1).detach()
    inputs.requires_grad = True
    model.requires_grad_(False)
    optimizer = torch.optim.Adam([inputs], lr=0.001)
    for i in range(300):
        p_marg = marginal_mog_log_prob(*ens.forward(torch.clamp(inputs, 0, 1)), x.reshape([-1, 1, 1]))
        # TODO: FIX to mean after exp
        pred = p_marg.mean(1)
        pred_d = pred[:, d]

        kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
        kl = torch.trapz(kl, x)
        #print(torch.argmin(kl,0), torch.argmax(kl, 0))
        print(kl.min(), kl.max())
        kl.mean().backward(retain_graph=True)
        optimizer.step()
    return torch.clamp(inputs[torch.argmin(kl)], 0, 1).detach()


def sample_attr(inits, d, goal_dist, x, grid, ens, w, goal_region, name, max_effort=100):
    inits = [traj for traj in inits if len(traj) > 5]

    def min_div(node, goal):
        node_feats = np.array(node.features, dtype=np.float)
        node_batch = torch.Tensor(node_feats).unsqueeze(0)
        # dists = ens.prob(node_batch, x)
        # dists = dists.T.reshape(-1, 120,120,120)
        # pred = ens.mean_prob(node_batch, x)
        p_marg = marginal_mog_log_prob(*ens.forward(node_batch), x.reshape([-1, 1, 1]))
        pred = p_marg.mean(1).squeeze()
        pred_d = pred[d]

        kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
        kl = torch.trapz(kl, x)
        # We want distribution to match, so drive kl to 0
        return kl.item()

    def batch_min_div(nodes, goal):
        if len(nodes) == 0:
            return np.array([])
        node_feats = np.array([node.features for node in nodes], dtype=np.float)
        cost = [trajectory_cost(goal_region, grid, node.trajectory) for node in nodes]
        node_batch = torch.Tensor(node_feats)
        # dists = ens.prob(node_batch, x)
        # dists = dists.T.reshape(-1, 120,120,120)
        # pred = ens.mean_prob(node_batch, x)
        p_marg = marginal_mog_log_prob(*ens.forward(node_batch), x.reshape([-1, 1, 1]))
        pred = p_marg.mean(1)
        pred_d = pred[:, d]

        kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
        kl = torch.trapz(kl, x)

        unc = ens_uncertainty_kl(ens, node_batch)
        together = np.vstack([unc.detach().numpy()[:, d].T, kl.detach().numpy(), cost])
        # We want distribution to match, so drive kl to 0
        scored = together[1] * w + together[2]
        print(
            f"{kl.min():.2f}, {together[2].min()} || {together[1][scored.argmin()]:.2f}, {together[2][scored.argmin()]} => {scored.min():.2f}")
        return scored.T

    def goal_kl(x_opt):
        mog = ens.forward(torch.Tensor(x_opt).unsqueeze(0))
        dist = marginal_mog_log_prob(*mog, torch.Tensor(x).reshape([-1, 1, 1])).mean(1).squeeze()
        pred_d = dist[d]
        kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
        kl = torch.trapz(kl, x)
        print(x_opt, kl.item())
        return kl.item()

    # best_in = scipy.optimize.minimize(goal_kl, np.full([11], .5), bounds=[(0,1) for _ in range(11)], method="l-bfgs-b", options={"eps":0.001})
    # best_in = scipy.optimize.basinhopping(goal_kl, np.full([11], .5), accept_test=MyBounds())

    modificiations = batch_hill_descend([TrajectoryNode(orig, None) for orig in inits],
                                        TrajectoryNode(ANY_PLAN, (None,)), grid,
                                        batch_min_div, max_tries=max_effort, tolerance=0.0, verbose=False)
    new = modificiations[-1].trajectory

    upper_bound_feats = optimize_feats(ens, goal_dist, d, x)
    print(upper_bound_feats)
    p_marg = marginal_mog_log_prob(*ens.forward(upper_bound_feats.reshape(-1, 11)), x.reshape([-1, 1, 1]))
    pred = p_marg.mean(1)
    pred_d = pred[:, d]
    kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
    kl = torch.trapz(kl, x).detach().numpy()
    pred_d = pred_d.exp().squeeze()
    fig = plt.figure()
    plt.plot(x, pred_d, label=f"ideal kl={kl}")
    new_feats = torch.tensor(TrajectoryNode.featurizer(new)).reshape([-1, 11]).float()
    p_marg = marginal_mog_log_prob(*ens.forward(new_feats), x.reshape([-1, 1, 1]))
    pred = p_marg.mean(1)
    pred_d = pred[:, d]
    kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
    kl = torch.trapz(kl, x).detach().numpy()
    pred_d = pred_d.exp().squeeze()
    plt.plot(x, pred_d, label=f"realized kl={kl}")
    plt.plot(x, goal_dist, label="goal")
    plt.legend()
    plt.title(name)
    return new


IG = 1000000
if __name__ == '__main__':
    grid, bedroom = load_map("interface/assets/house.tmx")

    _, demo_data, _, _ = process_turk_files(
        [f"data/{x}.csv" for x in ["pilot1", "active1", "active2", "mdn_active1", "mdn_active2"]])
    prompts = list(demo_data["aid"])
    demo_trajs = list(demo_data["traj"])
    demo_trajs = [traj for traj in demo_trajs if len(traj) > 0]
    demo_trajs = [traj for traj in demo_trajs if traj[-1] == (10, 9)]

    featurizer = partial(trajectory_features, bedroom, grid)
    TrajectoryNode.featurizer = featurizer

    ens = MDNEnsemble.load_ensemble(os.getcwd() + "/data/final")
    ens.freeze()

    x = np.linspace(-3, 3, 120)
    x_b = torch.Tensor(x)
    # x = np.mgrid[-3:3:0.05, -3:3:0.05, -3:3:0.05].reshape(3,-1).T
    goal_dist = torch.Tensor(normal(1, 0.3, x))
    inv_goal_dist = torch.Tensor(normal(-1, 0.3, x))

    """
    opt = astar(CoverageNode(bedroom, (10, 9)), CoverageNode([], (10, 9)), grid, coverage_manhattan)
    opt =  list(map(lambda x: x.point, opt))
    print("opt", trajectory_cost(bedroom, grid, opt), opt)
    better_opt = sample_attr([opt], 0, goal_dist, x_b, grid, ens, 10, bedroom)
    print("better", trajectory_cost(bedroom, grid, better_opt), better_opt)
    """
    opt = [(10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9), (17, 9), (18, 9), (19, 9), (19, 8), (20, 8),
           (20, 7), (20, 8), (20, 9), (21, 9), (21, 8), (21, 9), (21, 10), (22, 10), (22, 11), (22, 12), (23, 12),
           (23, 13), (22, 13), (21, 13), (20, 13), (20, 12), (21, 12), (21, 11), (22, 11), (23, 11), (24, 11), (24, 12),
           (24, 13), (25, 13), (25, 12), (25, 11), (25, 10), (25, 9), (26, 9), (26, 10), (26, 11), (26, 12), (27, 12),
           (27, 11), (27, 10), (27, 9), (27, 8), (26, 8), (25, 8), (25, 7), (25, 8), (25, 9), (25, 10), (24, 10),
           (23, 10), (22, 10), (21, 10), (20, 10), (20, 11), (20, 12), (19, 12), (19, 11), (19, 10), (19, 9), (18, 9),
           (17, 9), (16, 9), (15, 9), (14, 9), (13, 9), (12, 9), (11, 9), (10, 9)]
    print("TSK", trajectory_cost(bedroom, grid, opt))

    g0 = sample_attr([opt], 0, goal_dist, x_b, grid, ens, IG, bedroom, "In Competence B+")
    print("G0", trajectory_cost(bedroom, grid, g0), g0)

    g0_inv = sample_attr([opt], 0, inv_goal_dist, x_b, grid, ens, IG, bedroom, "In Competence B-")
    print("G0 inv", g0_inv)

    # g0_inv_bal = sample_attr([opt], 0, inv_goal_dist, x_b, grid, ens, 70.0, bedroom)
    # print("G0 inv bal", trajectory_cost(bedroom, grid, g0_inv_bal), g0_inv_bal)

    g0_bal = sample_attr([opt], 0, goal_dist, x_b, grid, ens, 200.0, bedroom, "In Competence BAL")
    print("G0 bal", trajectory_cost(bedroom, grid, g0_bal), g0_bal)

    g1 = sample_attr(demo_trajs, 1, goal_dist, x_b, grid, ens,IG, bedroom, "In Brokenness B+")
    print("G1", trajectory_cost(bedroom, grid, g1), g1)

    g1_inv = sample_attr(demo_trajs, 1, inv_goal_dist, x_b, grid, ens, IG, bedroom, "In Brokenness B-")
    print("G1 inv", trajectory_cost(bedroom, grid, g1_inv), g1_inv)

    g1_bal = sample_attr([opt], 1, goal_dist,x_b, grid, ens, 39.5, bedroom, "In Brokenness BAL", max_effort=100)
    print("G1 bal", trajectory_cost(bedroom, grid, g1_bal), g1_bal)

    g2 = sample_attr(demo_trajs, 2, goal_dist, x_b, grid, ens,IG, bedroom, "In Curiosity B+", max_effort=40)
    print("G2", trajectory_cost(bedroom, grid, g2), g2)

    g2_inv = sample_attr(demo_trajs, 2, inv_goal_dist,x_b, grid, ens, IG, bedroom, "In Curiosity B-",)
    print("G2 inv", trajectory_cost(bedroom, grid, g2_inv), g2_inv)

    g2_bal = sample_attr(demo_trajs, 2, goal_dist,x_b, grid, ens, 300.0, bedroom, "In Curiosity BAL")
    print("G2 bal", trajectory_cost(bedroom, grid, g2_bal), g2_bal)

    print("TEST DOMAIN")
    grid, bedroom = load_map("interface/assets/house_test.tmx")
    featurizer = partial(trajectory_features, bedroom, grid)
    TrajectoryNode.featurizer = featurizer
    """
    opt = astar(CoverageNode(bedroom, (20,10)), CoverageNode([], (20, 10)), grid, coverage_manhattan)
    opt =  list(map(lambda x: x.point, opt))
    print("opt", trajectory_cost(bedroom, grid, opt), opt)
    better_opt = sample_attr([opt], 0, goal_dist, x_b, grid, ens, 10, bedroom)
    print("better", trajectory_cost(bedroom, grid, better_opt), better_opt)
    """
    opt = [(20, 10), (19, 10), (18, 10), (17, 10), (16, 10), (15, 10), (15, 11), (14, 11), (14, 12), (14, 13), (14, 14),
           (13, 14), (13, 13), (12, 13), (12, 12), (13, 12), (13, 11), (12, 11), (12, 10), (12, 9), (13, 9), (14, 9),
           (14, 8), (14, 7), (13, 7), (13, 8), (12, 8), (12, 7), (12, 6), (11, 6), (11, 7), (11, 8), (11, 9), (10, 9),
           (10, 8), (10, 7), (9, 7), (9, 8), (9, 9), (9, 10), (8, 10), (8, 9), (8, 8), (8, 7), (7, 7), (6, 7), (6, 8),
           (7, 8), (7, 9), (7, 10), (7, 11), (6, 11), (6, 10), (6, 9), (5, 9), (5, 10), (5, 11), (4, 11), (4, 10),
           (4, 9), (3, 9), (3, 10), (3, 11), (3, 12), (4, 12), (5, 12), (6, 12), (7, 12), (7, 13), (8, 13), (8, 12),
           (8, 11), (9, 11), (9, 12), (9, 13), (10, 13), (10, 12), (10, 11), (11, 11), (11, 10), (12, 10), (13, 10),
           (14, 10), (15, 10), (16, 10), (17, 10), (18, 10), (19, 10), (20, 10)]
    print("TSK", trajectory_cost(bedroom, grid, opt))

    g0 = sample_attr([opt], 0, goal_dist, x_b, grid, ens,IG, bedroom, "Test Competence B+")
    print("G0", trajectory_cost(bedroom, grid, g0), g0)

    g0_inv = sample_attr([opt], 0, inv_goal_dist,x_b, grid, ens, IG, bedroom, "Test Competence B-")
    print("G0 inv", g0_inv)

    #g0_inv_bal = sample_attr([opt], 0, inv_goal_dist, x_b, grid, ens, 70.0, bedroom, "Test Competence BAL")
    #print("G0 inv bal", trajectory_cost(bedroom, grid, g0_inv_bal), g0_inv_bal)

    g0_bal = sample_attr([opt], 0, goal_dist,x_b, grid, ens, 400.0, bedroom, "Test Competence BAL")
    print("G0 bal", trajectory_cost(bedroom, grid, g0_bal), g0_bal)

    g1 = sample_attr([opt], 1, goal_dist, x_b, grid, ens,IG, bedroom, "Test Brokenness B+")
    print("G1", trajectory_cost(bedroom, grid, g1), g1)

    g1_inv = sample_attr([opt], 1, inv_goal_dist,x_b, grid, ens, IG, bedroom, "Test Brokenness B-")
    print("G1 inv", trajectory_cost(bedroom, grid, g1_inv), g1_inv)

    g1_bal = sample_attr([opt], 1, goal_dist,x_b, grid, ens, 84.0, bedroom, "Test Brokenness BAL", max_effort=200)
    print("G1 bal", trajectory_cost(bedroom, grid, g1_bal), g1_bal)

    g2 = sample_attr([opt], 2, goal_dist, x_b, grid, ens,IG, bedroom, "Test Curiosity B+", max_effort=70)
    print("G2", trajectory_cost(bedroom, grid, g2), g2)

    g2_inv = sample_attr([opt], 2, inv_goal_dist,x_b, grid, ens, IG, bedroom, "Test Curiosity B-")
    print("G2 inv", trajectory_cost(bedroom, grid, g2_inv), g2_inv)

    g2_bal = sample_attr([opt], 2, goal_dist,x_b, grid, ens, 250.0, bedroom, "Test Curiosity BAL")
    print("G2 bal", trajectory_cost(bedroom, grid, g2_bal), g2_bal)

    plt.show()
