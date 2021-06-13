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
from joblib import delayed, Parallel
from scipy.spatial.distance import cdist

from models.mdn.mdn import normal, marginal_mog_log_prob
from models.mdn.lit_mdn import LitMDN
from models.mdn.ensemble import MDNEnsemble, ens_uncertainty_kl
from models.util import gen_samples
from processing.loading import load_trajectory_pool, load_demo_pool, process_turk_files
from processing.mappings import attributions
from search.coverage import trajectory_features, trajectory_cost, CoverageNode, print_feats
from search.metric import coverage_manhattan
from search.routine import hill_descend, batch_hill_descend, astar
from search.sampling import sample_neighbors, sample_neighbors_for_pool
from search.trajectory import TrajectoryNode, ANY_PLAN
from search.util import load_map
import torch.nn.functional as F
import torch.optim.lr_scheduler

from matplotlib import pyplot as plt

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)


def optimize_feats(ens, goal_dist, d, x, max_effort=200):
    inputs = torch.rand((8192, 11), requires_grad=True)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    p_marg = marginal_mog_log_prob(*ens.forward(torch.clamp(inputs, 0, 1)), x.reshape([-1, 1, 1]))
    pred_d = p_marg[:,:,d].exp().mean(1).log()
    # inte = torch.trapz(pred_d.exp() * torch.linspace(0, 1, 120), x)

    kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
    kl = torch.trapz(kl, x)
    inputs = inputs.detach()[torch.topk(kl,64).indices]
    #inputs = inputs.detach()[torch.topk(inte,64).indices]
    inputs.requires_grad = True
    ens.requires_grad_(False)
    optimizer = torch.optim.Adam([inputs], lr=0.015)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.75)
    for i in range(max_effort + 1):
        optimizer.zero_grad()
        clamped_in = torch.clamp(inputs, 0, 1)
        p_marg = marginal_mog_log_prob(*ens.forward(clamped_in), x.reshape([-1, 1, 1]))
        pred_d = p_marg[:,:,d].exp().mean(1).log()

        kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
        kl = torch.trapz(kl, x)
        inte = -torch.trapz(pred_d.exp() * torch.linspace(0, 1, 120), x)

        if i % 50 == 0:
            print(i, torch.min(kl, 0).values.detach().numpy(), torch.min(inte, 0).values.detach().numpy())
        kl.mean().backward(retain_graph=True)
        #inte.mean().backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    raw_out = inputs[torch.argmin(kl)]
    return torch.clamp(raw_out, 0, 1).detach()


def sample_attr(inits, d, goal_dist, x, grid, ens, goal_region, max_effort=300, optimal_cost=None, cost_threshold=2):
    inits = [traj for traj in inits if len(traj) > 5]

    def batch_min_div(nodes, goal):
        if len(nodes) == 0:
            return np.array([])
        node_feats = np.array([node.features for node in nodes], dtype=np.float)
        cost = [node.G for node in nodes]
        node_batch = torch.Tensor(node_feats)
        # dists = ens.prob(node_batch, x)
        # dists = dists.T.reshape(-1, 120,120,120)
        # pred = ens.mean_prob(node_batch, x)
        p_marg = marginal_mog_log_prob(*ens.forward(node_batch), x.reshape([-1, 1, 1]))
        pred_d = p_marg[:,:,d].exp().mean(1).log()

        kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
        #inte = torch.trapz(pred_d * torch.linspace(0, 1, 120), x)
        kl = torch.trapz(kl, x)
        # uncertainty doesn't end up being that useful
        unc = ens_uncertainty_kl(ens, node_batch)
        together = np.vstack([unc.detach().numpy()[:, d].T, kl.detach().numpy(), cost])
        # Set a floor for task costs we'll allow
        together[2, :] /= optimal_cost
        together[2, together[2, :] > cost_threshold] = float('inf')
        # We want distribution to match, so drive kl to 0
        scored = together[1]
        #scored = -inte.detach().numpy()
        #print(f"kl={together[1][scored.argmin()]:.2f} c={together[2][scored.argmin()]:.2f}")
        return scored.T

    modificiations = batch_hill_descend([TrajectoryNode(orig, None, goal_region=goal_region, cost=trajectory_cost(goal_region, grid, orig)) for orig in inits],
                                        TrajectoryNode(ANY_PLAN, (None,), goal_region=goal_region), grid,
                                        batch_min_div, max_tries=max_effort, tolerance=float("-inf"), branch_limit=250, cost_bound=optimal_cost * cost_threshold, verbose=False)
    new = modificiations[-1].trajectory

    return new


def plot_lines(lines, x, goal, name):
    fig = plt.figure()
    cmap = plt.get_cmap('viridis')
    plt.plot(x, goal, label="goal", color="g")
    for label, (traj, line, cost, kl, ideal_gap) in lines.items():
        inte = torch.trapz(line * torch.linspace(0, 1, 120), x)
        if label == "Ideal":
            plt.plot(x, line, label=f"{label} kl={kl:.2f} i={inte:.2f}", color="b")
        else:
            label = float(label)
            color = ((label - 1) / 5) * .8 + .2
            color = min(color, 1.0)
            color = cmap(color)
            if label > 6:
                color = "orange"
            plt.plot(x, line, label=f"{label:.2f} kl={kl:.2f} c={cost} g={ideal_gap:.2f} i={inte:.2f}", color=color)

    plt.legend()
    plt.title(name)
    return fig


def feats_to_density(feats, ens, x, d):
    if not isinstance(feats, torch.Tensor):
        feats = torch.Tensor(feats)
    p_marg = marginal_mog_log_prob(*ens.forward(feats.reshape(-1, 11)), x.reshape([-1, 1, 1]))
    pred_d = p_marg[:,:,d].exp().mean(1)
    pred_d = pred_d.squeeze()
    return pred_d


def sweep_constraint(start_traj, goal_dist, d, grid, goal_region, x, ens, opt_cost, name):
    vals = [1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 10]

    def get_traj_for_suboptimality(val):
        featurizer = partial(trajectory_features, goal_region, grid)
        TrajectoryNode.featurizer = featurizer
        return sample_attr([start_traj], d, goal_dist, x, grid, ens, goal_region,
                                    optimal_cost=opt_cost, cost_threshold=val)
    trajs = Parallel(n_jobs=len(vals),verbose=0)(delayed(get_traj_for_suboptimality)(val) for val in vals)
    costs = [trajectory_cost(goal_region, grid, traj) for traj in trajs]

    print(name, "******************************************************")
    lines = {str(val): (traj, cost) for val, traj, cost in zip(vals, trajs, costs)}
    upper_bound_feats = optimize_feats(ens, goal_dist, d, x)
    ideal_line = feats_to_density(upper_bound_feats, ens, x, d)
    kl = F.kl_div(ideal_line.log(), goal_dist, reduction='none', log_target=False)
    ideal_kl = torch.trapz(kl, x)

    for key, (traj, cost) in lines.items():
        traj_feats = TrajectoryNode.featurizer(traj)
        line = feats_to_density(traj_feats, ens, x, d)
        kl = F.kl_div(line.log(), goal_dist, reduction='none', log_target=False)
        kl = torch.trapz(kl, x)
        ideal_gap = np.linalg.norm(traj_feats - upper_bound_feats.detach().numpy(), 1)
        lines[key] = (traj, line, cost, kl, ideal_gap)
        print(f"{key}: c={cost} kl={kl:.2f} gap={ideal_gap:.2f}")
        print_feats(traj_feats)
        print(traj)
        print()

    lines["Ideal"] = (None, ideal_line, None, ideal_kl, None)
    print(f"Ideal: kl={ideal_kl:.2f}")
    print_feats(upper_bound_feats)
    print()

    line_fig = plot_lines(lines, x, goal_dist, name)


def main():
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
    goal_dist = torch.Tensor(normal(1.5, 0.3, x))
    inv_goal_dist = torch.Tensor(normal(-1.5, 0.3, x))

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
    opt_cost = trajectory_cost(bedroom, grid, opt)

    sweep_constraint(opt, goal_dist, 0, grid, bedroom, x_b, ens, opt_cost, "In Competence B+")
    sweep_constraint(opt, inv_goal_dist, 0, grid, bedroom, x_b, ens, opt_cost, "In Competence B-")
    sweep_constraint(opt, goal_dist, 1, grid, bedroom, x_b, ens, opt_cost, "In Brokenness B+")
    sweep_constraint(opt, inv_goal_dist, 1, grid, bedroom, x_b, ens, opt_cost, "In Brokenness B-")
    sweep_constraint(opt, goal_dist, 2, grid, bedroom, x_b, ens, opt_cost, "In Curious B+")
    sweep_constraint(opt, inv_goal_dist, 2, grid, bedroom, x_b, ens, opt_cost, "In Curious B-")

    print("TSK", trajectory_cost(bedroom, grid, opt))

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
    opt_cost = trajectory_cost(bedroom, grid, opt)

    sweep_constraint(opt, goal_dist, 0, grid, bedroom, x_b, ens, opt_cost, "Test Competence B+")
    sweep_constraint(opt, inv_goal_dist, 0, grid, bedroom, x_b, ens, opt_cost, "Test Competence B-")
    sweep_constraint(opt, goal_dist, 1, grid, bedroom, x_b, ens, opt_cost, "Test Brokenness B+")
    sweep_constraint(opt, inv_goal_dist, 1, grid, bedroom, x_b, ens, opt_cost, "Test Brokenness B-")
    sweep_constraint(opt, goal_dist, 2, grid, bedroom, x_b, ens, opt_cost, "Test Curious B+")
    sweep_constraint(opt, inv_goal_dist, 2, grid, bedroom, x_b, ens, opt_cost, "Test Curious B-")
    plt.show()


if __name__ == '__main__':
    main()
