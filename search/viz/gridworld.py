import pylab as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.cm as cm

from matplotlib.text import OffsetFrom
from matplotlib.patches import Arrow


def plot_num(ax, xy, offset, value, **kwargs):
    x, y = xy
    offset = offset * .47

    textxy = (x + offset[0] * .85, y + offset[1] * .85)

    ax.annotate(value,
                xy=(x, y), xycoords='data',
                xytext=textxy,
                horizontalalignment='center',
                verticalalignment='center', **kwargs)


def plot_arrow(ax, xy, offset, color):
    x, y = xy
    offset = offset * .47

    arrowx = x + offset[0] * .65
    arrowy = y + offset[1] * .65
    ax.add_patch(
        Arrow(arrowx, arrowy, offset[0] * .35, offset[1] * .35, width=.5, facecolor=color))


def plot_square(ax, xy, offset, color):
    x, y = xy
    offset = offset * .47

    x = x + offset[0] * .15
    y = y + offset[1] * .15

    ax.add_patch(
        patches.Rectangle(
            (x - .05, y - .05),
            .1,  # width
            .1,  # height
            facecolor=color,
            edgecolor='black'
        )
    )


def color_square(ax, xy, color):
    x, y = xy
    ax.add_patch(
        patches.Rectangle(
            (x - .5, y - .5),
            .99,  # width
            .99,  # height
            color=color,
            alpha=.5
        )
    )


def plot_labeled_arrows(dirs):
    def draw(ax, xy, qs):

        color_square(ax, xy, cm.coolwarm(max(qs)))

        for q, direction in zip(qs, dirs):
            v = (q - min(qs)) / (max(qs) - min(qs))
            if not all(direction == CENTER):
                plot_arrow(ax, xy, direction, cm.coolwarm(v))
            else:
                plot_square(ax, xy, direction, cm.coolwarm(v))

            plot_num(ax, xy, direction, "%.3g" % q)

    return draw


def rowcol_to_xy(gridworld, rowcol):
    row, col = rowcol
    return col - 1 + .5, row + .5,


CENTER = np.array((0., 0))
LEFT = np.array((-1., 0))
RIGHT = np.array((1., 0))
UP = np.array((0., 1))
DOWN = np.array((0., -1))

action_to_arrow = [RIGHT, UP, LEFT, DOWN]


def compute_vals(qs):
    return [max(qlist) for qlist in qs.values()]


def normalize_q(Qs):
    # vals = compute_vals(Qs)
    allq = [q for qs in Qs.values() for q in qs]
    # vmax, vmin = max(vals), min(vals)
    vmax, vmin = max(allq), min(allq)
    range = max(vmax - vmin, 1e-8)

    def norm(q):
        return (q - vmin) / range

    return {s: map(norm, qs) for s, qs in Qs.iteritems()}


def plotQ(ax, gridworld, qs, plot_square):
    qs = normalize_q(qs)

    for s in range(gridworld.nState):
        rc = gridworld.row_and_column(s)
        xy = rowcol_to_xy(gridworld, rc)

        plot_square(ax, xy, qs[s])


def plotR(ax, gridworld, r):
    for s in range(gridworld.n_states):
        rs = r[s]

        rc = gridworld.int_to_point(s)
        xy = rowcol_to_xy(gridworld, rc)

        plot_num(ax, xy, (UP + RIGHT) * .5, "r=%.3g" % rs, alpha=.5)


def plotRBelief(ax, gridworld, Rs):
    for s in range(gridworld.nState):
        m, precision = Rs[s, 0]
        sd = precision ** -.5

        rc = gridworld.row_and_column(s)
        xy = rowcol_to_xy(gridworld, rc)

        plot_num(ax, xy, (.5 * UP + RIGHT) * .5, "[%.3g, %.3g]" % (m - sd, m + sd))


def grid_plot(gridworld, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.set_xticks(range(gridworld.grid_size + 1))
    ax.set_yticks(range(gridworld.grid_size + 1))
    plt.grid(True)
    return ax


def plot_traj(ax, gridworld, traj):
    traj_int = traj.astype(int)
    for s, a, _ in traj_int:
        xy = gridworld.int_to_point(s)
        plot_arrow(ax, (xy[0] + .5, xy[1] + .5), action_to_arrow[a], (0, 0, 0))
    return ax


def show_reward_function(ground_truth, recovered):
    plt.subplot(1, 2, 1)
    plt.pcolor(ground_truth)
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(recovered)
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

def show_trajectories(gw, traj, traj2):
    ax = plt.subplot(1, 2, 1)
    grid_plot(gw, ax)
    plot_traj(ax, gw, traj)
    ax = plt.subplot(1,2,2)
    grid_plot(gw, ax)
    plot_traj(ax, gw, traj2)
    plt.gcf().show()