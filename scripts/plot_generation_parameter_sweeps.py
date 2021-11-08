import ast
import os

import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

from models.mdn.ensemble import MDNEnsemble
from models.mdn.mdn import normal, marginal_mog_log_prob
from processing.mappings import factor_names
from search.coverage import print_feats


def feats_to_density(feats, ens, x, d):
    if not isinstance(feats, torch.Tensor):
        feats = torch.Tensor(feats)
    p_marg = marginal_mog_log_prob(*ens.forward(feats.reshape(-1, 11)), x.reshape([-1, 1, 1]))
    pred_d = p_marg[:, :, d].exp().mean(1)
    pred_d = pred_d.squeeze()
    return pred_d


def print_res(res):
    for i, (traj, features, cost, floor, kl) in res.iterrows():
        print(f"Subpotimality: {floor} kl: {kl} cost: {cost}")
        print_feats(features)


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def main():
    ens = MDNEnsemble.load_ensemble(os.getcwd() + "/data/final_new_feats")
    ens.freeze()

    plt.rcParams["font.family"] = "Times New Roman"
    SMALL_SIZE = 7
    BIGGER_SIZE = 8

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('lines', markersize=1.5)  # fontsize of the figure title
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(5.5, 3.0))

    x = np.linspace(-3, 3, 120)
    x_b = torch.Tensor(x)
    goal_dist = torch.Tensor(normal(1.5, 0.3, x))

    colors = plt.get_cmap("Dark2").colors

    for i, factor_name in enumerate(factor_names):
        name = f"in_{factor_name}_pos"
        pos_res = pd.read_csv(name + ".csv", index_col=0)
        # Drop the manually placed ones
        pos_res = pos_res[~pos_res["floor"].isin([2, 4, 8, 12, 16])]
        print(name)
        ax = axs[0][i]
        cmap = plt.get_cmap('viridis')
        ax.plot(x, goal_dist, label="Goal", linewidth=0.75, color=colors[2])
        for j, (traj, features, cost, floor, kl) in pos_res.iterrows():
            features = ast.literal_eval(features)
            line = feats_to_density(features, ens, x_b, i)
            if np.isnan(floor) and np.isnan(cost):
                label = "Bound"
                color = colors[1]
                ideal_kl = kl
            elif np.isnan(floor):
                continue
            else:
                label = None
                color = np.log10(floor) / np.log10(pos_res["floor"].max())
                color = cmap(color)

            ax.plot(x, line, label=label, color=color, linewidth=0.75)

        ax.set_title(factor_name.capitalize())

        ax.set_ylim((-0.05, 1.55))
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        if i == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_yticklabels([])

        if i == 2:
            ax.legend()
        ax.set_xlabel("Factor score")

        ax = axs[1][i]
        ax.set_ylim((0, 4.05))

        ax.set_xscale("log", basex=2)
        ax.set_xlabel("Suboptimality")
        ax.set_xticks([1, 2 ** 1, 2 ** 2, 2 ** 2, 2 ** 3])
        ax.set_yticks([0, 1, 2, 3, 4])
        if i == 0:
            ax.set_ylabel("KL divergence")
        else:
            ax.set_yticklabels([])

        points = np.array([pos_res["floor"], pos_res["kl"]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, np.log10(pos_res["floor"].max()))
        # lc = LineCollection(segments, cmap='PiYG', norm=norm)
        lc = LineCollection(segments, cmap='magma', norm=norm)
        # Set the values used for colormapping
        lc.set_array(np.log10(pos_res["floor"]))
        lc.set_linewidth(2)
        # line = ax.add_collection(lc)
        ax.axhline(ideal_kl, label="Bound", color=colors[1], linewidth=0.75)
        ax.scatter(pos_res["floor"], pos_res["kl"], c=np.log10(pos_res["floor"]) / np.log10(pos_res["floor"].max()),
                   cmap='viridis')

    with PdfPages(f'sweep-I.pdf') as pp:
        fig.tight_layout(h_pad=0, w_pad=0.5)
        pp.savefig(fig, bbox_inches='tight', transparent="True", pad_inches=0)

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(5.5, 3.0))
    pos_by_factor = []
    for i, factor_name in enumerate(factor_names):
        name = f"test_{factor_name}_pos"
        pos_res = pd.read_csv(name + ".csv", index_col=0)
        # Drop the manually placed ones
        pos_res = pos_res[~pos_res["floor"].isin([2, 4, 8, 12, 16])]
        print(name)
        ax = axs[0][i]
        cmap = plt.get_cmap('viridis')
        ax.plot(x, goal_dist, label="Goal", linewidth=0.75, color=colors[2])
        for j, (traj, features, cost, floor, kl) in pos_res.iterrows():
            features = ast.literal_eval(features)
            line = feats_to_density(features, ens, x_b, i)
            if np.isnan(floor) and np.isnan(cost):
                label = "Bound"
                color = colors[1]
                ideal_kl = kl
            elif np.isnan(floor):
                continue
            else:
                label = None
                color = np.log10(floor) / np.log10(pos_res["floor"].max())
                color = cmap(color)

            ax.plot(x, line, label=label, color=color, linewidth=0.75)

        ax.set_title(factor_name.capitalize())

        ax.set_ylim((-0.05, 1.55))
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        if i == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_yticklabels([])

        if i == 2:
            ax.legend()
        ax.set_xlabel("Factor score")

        ax = axs[1][i]
        ax.set_ylim((0, 4.05))

        ax.set_xscale("log", basex=2)
        ax.set_xlabel("Suboptimality")
        ax.set_xticks([1, 2 ** 1, 2 ** 2, 2 ** 2, 2 ** 3])
        ax.set_yticks([0, 1, 2, 3, 4])
        if i == 0:
            ax.set_ylabel("KL divergence")
        else:
            ax.set_yticklabels([])

        points = np.array([pos_res["floor"], pos_res["kl"]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, np.log10(pos_res["floor"].max()))
        # lc = LineCollection(segments, cmap='PiYG', norm=norm)
        lc = LineCollection(segments, cmap='magma', norm=norm)
        # Set the values used for colormapping
        lc.set_array(np.log10(pos_res["floor"]))
        lc.set_linewidth(2)
        # line = ax.add_collection(lc)
        ax.axhline(ideal_kl, label="Bound", color=colors[1], linewidth=0.75)
        ax.scatter(pos_res["floor"], pos_res["kl"], c=np.log10(pos_res["floor"]) / np.log10(pos_res["floor"].max()),
                   cmap='viridis')

    with PdfPages(f'sweep-II.pdf') as pp:
        fig.tight_layout(h_pad=0, w_pad=0.5)
        pp.savefig(fig, bbox_inches='tight', transparent="True", pad_inches=0)


if __name__ == '__main__':
    main()
