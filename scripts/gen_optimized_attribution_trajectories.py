"""
Generates trajectories optimized to elicit a particular attribution.
This script sweeps the suboptimality constraint and dumps the
trajectories to a file for further analysis and plot-making.
Trajectories used in our experiments came from these procedures.
"""
import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler
from joblib import delayed, Parallel
from matplotlib import pyplot as plt

from models.mdn.ensemble import MDNEnsemble, ens_uncertainty_kl
from models.mdn.mdn import normal, marginal_mog_log_prob
from processing.loading import process_turk_files
from search.coverage import trajectory_features, trajectory_cost, print_feats
from search.routine import batch_hill_descend
from search.trajectory import TrajectoryNode, ANY_PLAN
from search.util import load_map
from processing.mappings import factor_names
import shelve


def shelve_it(file_name):
    d = shelve.open(file_name, protocol=5)

    def decorator(func):
        def new_func(*args):
            key = repr(tuple(args))
            if key not in d:
                d[key] = func(*args)
            return d[key]

        return new_func

    return decorator


random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)


@shelve_it("optimized_feats")
def optimize_feats(ens, goal_dist, d, x, max_effort):
    """
    Perform gradient descent on a vector of input features, stepping
    in the direction of minimizing KL divergence with a desired
    attribution. Note that the features will almost surely not
    be realizable with a feasible trajectory.
    :param ens: model
    :param goal_dist: the desired attribution
    :param d: the dimension of the attribution model to optimize (one attribution at a time in this example)
    :param x: domain to evaluate attribution over
    :param max_effort:
    :return:
    """
    inputs = torch.rand((8192, 11), requires_grad=True)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    p_marg = marginal_mog_log_prob(*ens.forward(torch.clamp(inputs, 0, 1)), x.reshape([-1, 1, 1]))
    pred_d = p_marg[:, :, d].exp().mean(1).log()
    # inte = torch.trapz(pred_d.exp() * torch.linspace(0, 1, 120), x)

    kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
    kl = torch.trapz(kl, x)
    inputs = inputs.detach()[torch.topk(kl, 64).indices]
    # inputs = inputs.detach()[torch.topk(inte,64).indices]
    inputs.requires_grad = True
    ens.requires_grad_(False)
    optimizer = torch.optim.Adam([inputs], lr=0.015)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.75)
    for i in range(max_effort + 1):
        optimizer.zero_grad()
        clamped_in = torch.clamp(inputs, 0, 1)
        p_marg = marginal_mog_log_prob(*ens.forward(clamped_in), x.reshape([-1, 1, 1]))
        pred_d = p_marg[:, :, d].exp().mean(1).log()

        kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
        kl = torch.trapz(kl, x)
        inte = -torch.trapz(pred_d.exp() * torch.linspace(0, 1, 120), x)

        if i % 25 == 0:
            print(i, torch.min(kl, 0).values.detach().numpy(), torch.min(inte, 0).values.detach().numpy())
        kl.mean().backward(retain_graph=True)
        # inte.mean().backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    raw_out = inputs[torch.argmin(kl)]
    return torch.clamp(raw_out, 0, 1).detach().numpy()


def search_for_attribution_trajectory(initializations, d, goal_dist, x, grid, ens, goal_region, max_effort=750, optimal_cost=None, allowed_suboptimality=None):
    """
    Search in the space of trajectories towards a trajectory that minimizes KL divergence
    with a goal attribution distribution
    :param initializations: trajectories to start searching from
    :param d: attribution dimension to optimize
    :param goal_dist: desired attribution distribution
    :param x: domain to optimize attribution over
    :param grid: the environment for the trajectory
    :param ens: the model to optimize against
    :param goal_region: the task specification
    :param max_effort:
    :param optimal_cost: best trajectory cost for the task, so we can enforce a constraint if desired
    :param allowed_suboptimality: in percentage of optimal cost
    :return:
    """
    initializations = [traj for traj in initializations if len(traj) > 5]

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
        pred_d = p_marg[:, :, d].exp().mean(1).log()

        kl = F.kl_div(pred_d, goal_dist, reduction='none', log_target=False)
        # inte = torch.trapz(pred_d * torch.linspace(0, 1, 120), x)
        kl = torch.trapz(kl, x)
        # uncertainty doesn't end up being that useful
        unc = ens_uncertainty_kl(ens, node_batch)
        together = np.vstack([unc.detach().numpy()[:, d].T, kl.detach().numpy(), cost])
        # Set a floor for task costs we'll allow
        together[2, :] /= optimal_cost
        together[2, together[2, :] > allowed_suboptimality] = float('inf')
        # We want distribution to match, so drive kl to 0
        scored = together[1]
        # scored = -inte.detach().numpy()
        # print(f"kl={together[1][scored.argmin()]:.2f} c={together[2][scored.argmin()]:.2f}")
        return scored.T

    modifications = batch_hill_descend(
        [TrajectoryNode(orig, None, goal_region=goal_region, cost=trajectory_cost(goal_region, grid, orig)) for orig in
         initializations],
        TrajectoryNode(ANY_PLAN, (None,), goal_region=goal_region), grid,
        batch_min_div, max_tries=max_effort, tolerance=float("-inf"), branch_limit=250,
        cost_bound=optimal_cost * allowed_suboptimality, verbose=False)
    new = modifications[-1].trajectory

    return new


def polish_trajectory(inits, grid, goal_region, max_effort=300):
    """
    Use our trajectory search to look for improvements to a trajectory.
    :param inits:
    :param grid:
    :param goal_region:
    :param max_effort:
    :return:
    """
    def batch_cost(nodes, goal):
        if len(nodes) == 0:
            return np.array([])
        cost = [node.G for node in nodes]
        return np.array(cost)

    modificiations = batch_hill_descend(
        [TrajectoryNode(orig, None, goal_region=goal_region, cost=trajectory_cost(goal_region, grid, orig)) for orig in
         inits],
        TrajectoryNode(ANY_PLAN, (None,), goal_region=goal_region), grid,
        batch_cost, max_tries=max_effort, tolerance=float("-inf"), branch_limit=250, verbose=False)
    new = modificiations[-1].trajectory

    return new


def feats_to_density(feats, ens, x, d):
    if not isinstance(feats, torch.Tensor):
        feats = torch.Tensor(feats)
    p_marg = marginal_mog_log_prob(*ens.forward(feats.reshape(-1, 11)), x.reshape([-1, 1, 1]))
    pred_d = p_marg[:, :, d].exp().mean(1)
    pred_d = pred_d.squeeze()
    return pred_d


def sweep_suboptimality_constraint(sweep_vals, start_traj, goal_dist, d, grid, goal_region, x, ens, opt_cost, upper_bound_feats):
    """
    Searches for optimized trajectories while sweeping the allowable task
    suboptimality
    :param sweep_vals:
    :param start_traj:
    :param goal_dist:
    :param d:
    :param grid:
    :param goal_region:
    :param x:
    :param ens:
    :param opt_cost:
    :param upper_bound_feats:
    :return:
    """
    sweep_results = []

    def get_traj_for_suboptimality(val):
        featurizer = partial(trajectory_features, goal_region, grid)
        TrajectoryNode.featurizer = featurizer
        return search_for_attribution_trajectory([start_traj], d, goal_dist, x, grid, ens, goal_region,
                                                 optimal_cost=opt_cost, allowed_suboptimality=val)

    trajs = Parallel(n_jobs=20, verbose=0)(delayed(get_traj_for_suboptimality)(val) for val in sweep_vals)
    costs = [trajectory_cost(goal_region, grid, traj) for traj in trajs]

    lines = {str(val): (traj, cost) for val, traj, cost in zip(sweep_vals, trajs, costs)}

    start_feats = TrajectoryNode.featurizer(start_traj)
    start_line = feats_to_density(start_feats, ens, x, d)
    kl = F.kl_div(start_line.log(), goal_dist, reduction='none', log_target=False)
    start_kl = torch.trapz(kl, x).item()
    sweep_results.append((start_traj, start_feats, opt_cost, None, start_kl))

    ideal_line = feats_to_density(upper_bound_feats, ens, x, d)
    kl = F.kl_div(ideal_line.log(), goal_dist, reduction='none', log_target=False)
    ideal_kl = torch.trapz(kl, x).item()
    sweep_results.append((None, upper_bound_feats, None, None, ideal_kl))

    for key, (traj, cost) in lines.items():
        traj_feats = TrajectoryNode.featurizer(traj)
        line = feats_to_density(traj_feats, ens, x, d)
        kl = F.kl_div(line.log(), goal_dist, reduction='none', log_target=False)
        kl = torch.trapz(kl, x).item()
        sweep_results.append((traj, traj_feats, cost, float(key), kl))

    return pd.DataFrame(data=sweep_results, columns=["trajectory", "features", "cost", "floor", "kl"])

def print_results(res):
    for i, (traj, features, cost, floor, kl) in res.iterrows():
        print(f"Subpotimality: {floor} kl: {kl} cost: {cost}")
        print_feats(features)


def main():
    grid, bedroom = load_map("interface/assets/house.tmx")

    _, demo_data, _, _ = process_turk_files(
        [f"data/{x}.csv" for x in ["pilot1", "active1", "active2", "mdn_active1", "mdn_active2"]])
    prompts = list(demo_data["aid"])

    featurizer = partial(trajectory_features, bedroom, grid)
    TrajectoryNode.featurizer = featurizer

    ens = MDNEnsemble.load_ensemble(os.getcwd() + "/data/final_new_feats")
    ens.freeze()

    x = np.linspace(-3, 3, 120)
    x_b = torch.Tensor(x)
    # x = np.mgrid[-3:3:0.05, -3:3:0.05, -3:3:0.05].reshape(3,-1).T
    goal_dist = torch.Tensor(normal(1.5, 0.3, x))

    """opt = astar(CoverageNode(bedroom, (10, 9)), CoverageNode([], (10, 9)), grid, coverage_manhattan)
    opt = list(map(lambda x: x.point, opt))
    print("opt", trajectory_cost(bedroom, grid, opt), opt)
    # Heuristic might not be correct, so let's use shortcutting to double check that we're getting the lowest cost
    better_opt = polish_trajectory([opt], grid, bedroom)
    print("better", trajectory_cost(bedroom, grid, better_opt), better_opt)"""

    opt = [(10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9), (17, 9), (18, 9), (19, 9), (19, 8), (20, 8),
           (20, 7), (20, 8), (21, 8), (21, 9), (20, 9), (19, 9), (19, 10), (19, 11), (20, 11), (21, 11), (22, 11),
           (23, 11), (23, 12), (22, 12), (21, 12), (20, 12), (19, 12), (20, 12), (20, 13), (21, 13), (22, 13), (23, 13),
           (24, 13), (25, 13), (25, 12), (24, 12), (24, 11), (25, 11), (26, 11), (26, 12), (27, 12), (27, 11), (27, 10),
           (26, 10), (26, 9), (27, 9), (27, 8), (26, 8), (25, 8), (25, 7), (25, 8), (25, 9), (25, 10), (24, 10),
           (23, 10), (22, 10), (21, 10), (20, 10), (19, 10), (18, 10), (17, 10), (16, 10), (15, 10), (15, 9), (14, 9),
           (13, 9), (12, 9), (11, 9), (10, 9)]
    opt_cost = trajectory_cost(bedroom, grid, opt)
    print(opt, opt_cost, opt)

    pos_upper_bounds = [optimize_feats(ens, goal_dist, i, x, 200) for i in range(3)]

    vals = np.logspace(0, 1, 20)
    # Ensure we hit the values used in our experiments
    vals = np.insert(vals, np.searchsorted(vals, [2, 4, 8, 12, 16]), [2, 4, 8, 12, 16])
    pos_by_factor = []
    for i, factor_name in enumerate(factor_names):
        name = f"in_{factor_name}_pos"
        print(name)
        pos_res = sweep_suboptimality_constraint(vals, opt, goal_dist, i, grid, bedroom, x_b, ens, opt_cost, pos_upper_bounds[i])
        pos_res.to_csv(name + ".csv")
        print_results(pos_res)
        pos_by_factor.append(pos_res)

    trajs = []
    picked = [1, 2, 4, 12]
    for res in pos_by_factor:
        for v in picked:
            trajs.append(res[res["floor"] == v]["trajectory"].values[0])
    print(trajs)

    grid, bedroom = load_map("interface/assets/house_test.tmx")
    featurizer = partial(trajectory_features, bedroom, grid)
    TrajectoryNode.featurizer = featurizer

    """opt = astar(CoverageNode(bedroom, (20,10)), CoverageNode([], (20, 10)), grid, coverage_manhattan)
    opt = list(map(lambda x: x.point, opt))
    print("opt", trajectory_cost(bedroom, grid, opt), opt)
    better_opt = polish_trajectory([opt], grid, bedroom)
    print("better", trajectory_cost(bedroom, grid, better_opt), better_opt)"""

    opt = [(20, 10), (19, 10), (18, 10), (17, 10), (16, 10), (15, 10), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14),
           (13, 14), (13, 13), (12, 13), (12, 12), (13, 12), (13, 11), (12, 11), (12, 10), (12, 9), (13, 9), (14, 9),
           (14, 8), (14, 7), (13, 7), (13, 8), (12, 8), (12, 7), (12, 6), (13, 6), (12, 6), (11, 6), (11, 7), (11, 8),
           (11, 9), (10, 9), (10, 8), (10, 7), (9, 7), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (8, 10), (8, 9), (8, 8),
           (8, 7), (7, 7), (6, 7), (6, 8), (7, 8), (7, 9), (7, 10), (7, 11), (6, 11), (6, 10), (6, 9), (5, 9), (5, 10),
           (5, 11), (4, 11), (4, 10), (4, 9), (3, 9), (3, 10), (3, 11), (3, 12), (4, 12), (5, 12), (6, 12), (7, 12),
           (7, 13), (8, 13), (8, 12), (8, 11), (9, 11), (9, 12), (9, 13), (9, 14), (9, 13), (10, 13), (10, 12),
           (10, 11), (11, 11), (11, 10), (10, 10), (11, 10), (12, 10), (13, 10), (14, 10), (15, 10), (16, 10), (17, 10),
           (18, 10), (19, 10), (20, 10)]
    opt_cost = trajectory_cost(bedroom, grid, opt)
    print(opt, opt_cost, opt)

    pos_by_factor = []
    for i, factor_name in enumerate(factor_names):
        name = f"test_{factor_name}_pos"
        print(name)
        pos_res = sweep_suboptimality_constraint(vals, opt, goal_dist, i, grid, bedroom, x_b, ens, opt_cost, pos_upper_bounds[i])
        pos_res.to_csv(name + ".csv")
        print_results(pos_res)
        pos_by_factor.append(pos_res)

    trajs = []
    picked = [1, 2, 4, 12]
    for res in pos_by_factor:
        for v in picked:
            trajs.append(res[res["floor"] == v]["trajectory"].values[0])
    print(trajs)


if __name__ == '__main__':
    main()
