import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def plot_components(components, ax, question_names, condition_name, title_prefix=""):
    loading_mat = components
    num_factors = loading_mat.shape[1]
    ax.set_yticks(np.arange(len(question_names)))
    ax.set_xticks(np.arange(num_factors))
    # ... and label them with the respective list entries
    ax.set_yticklabels(question_names)
    ax.set_title(title_prefix + condition_name)
    im = ax.imshow(loading_mat)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(question_names)):
        for j in range(num_factors):
            text = ax.text(j, i, "{:.2f}".format(loading_mat[i, j]),
                           ha="center", va="center", color="w", fontsize=8)


def make_conf_mat_plots(clfs, x, y):
    fig, axs = plt.subplots(len(clfs), 1, figsize=(8.5, 11))
    fig.suptitle("Attribution Confusion Matrices")
    i = 0
    for clf in clfs:
        row = i
        ax = axs[row]
        ax.set_title(["competent", "broken", "curious"][i])
        plot_confusion_matrix(clf, x, y[i], ax=ax, display_labels=["Disagree", "Neither", "Agree"])
        i += 1
    fig.subplots_adjust(hspace=.5)
    return fig


def plot_histograms(title, question_names, data, upper_bound=None):
    i = 0
    fig, axs = plt.subplots((len(question_names) + 1) // 2, 2, figsize=(8.5, 11))
    fig.suptitle(title)
    for question in question_names:
        row = math.floor(i / 2)
        column = i % 2
        ax = axs[row][column]
        ax.set_title(question)
        if upper_bound:
            ax.set_ylim((0, upper_bound))
            ax.set_yticks(np.arange(0, upper_bound, 1))
            if upper_bound > 8:
                ax.set_yticks(np.linspace(0, upper_bound, 9))
        ax.grid(axis="y")
        q_data = data[question]
        # as_array = question.to_numpy(dtype=np.int).flatten()
        as_dict = q_data.append(pd.Series([1, 2, 3, 4, 5]), ignore_index=True).value_counts()
        as_dict = {key: value - 1 for key, value in as_dict.items()}
        values = [value for _, value in sorted(as_dict.items())]

        ax.bar([1, 2, 3, 4, 5], values)
        i += 1
    return plt.gcf()


def make_scatterplot(feature_names, question, data):
    fig, axs = plt.subplots((len(feature_names) + 1) // 2, 2, figsize=(8.5, 11))
    # fig.suptitle("Feature vs attribution {}".format(question))
    for i, feat_name in enumerate(feature_names):
        row = math.floor(i / 2)
        column = i % 2
        ax = axs[row][column]
        ax.set_title("{} vs {}".format(feat_name, question))
        ax.set_ylim((0, 6))
        ax.set_yticks([1, 2, 3, 4, 5])

        ax.set_xlim((0, 1))
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.grid(axis="both")
        q_data = data[question]
        feat_data = data["features"].apply(lambda row: row[i])
        ax.plot(feat_data, q_data, "bo")
    return plt.gcf()


def attribution_scatter(attrs):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*attrs.T.to_numpy(), marker="o")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    ax.set_xlabel('Competence')
    ax.set_ylabel('Brokeness')
    ax.set_zlabel('Curiosity')

    return fig


def make_fa_plots(data, analysis, question_names):
    # Run factor analysis by condition
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
    plot_components(analysis.loadings_, ax, question_names, "All", title_prefix="FA ")

    """
    # scikit's FA doesn't have rotations, so it's harder to interpret
    transformer = FactorAnalysis(num_factors)
    analysis = transformer.fit(group.to_numpy())
    plot_components(analysis.components_.transpose(), title_prefix="FA ")
    """
    return fig


def make_density(name, data, true_points=None):
    from scipy.stats import gaussian_kde
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    for i, column in enumerate(data.columns):
        density = gaussian_kde(data[column].tolist())
        xs = np.linspace(-3, 3, 200)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        plt.plot(xs, density(xs), "-", label=["competent", "broken", "curious"][i])
    if true_points is not None:
        for i in range(true_points.shape[1]):
            plt.scatter(true_points[:, i], np.zeros(len(true_points)) + i * 0.05, marker="x")
    ax.legend()
    ax.set_title(name)
    ax.set_xlabel("Factor Scores")
    ax.set_ylim([-0.05, 1.4])
    return fig


def make_mog(name, probs, true_points=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    x = np.linspace(-3, 3, 200)
    for d in range(probs.shape[0]):
        mog_p = probs[d]
        tot = np.trapz(mog_p, x)
        plt.plot(x, mog_p, "-",
                 label=["competent", "broken", "curious"][d])
    if true_points is not None:
        for i in range(true_points.shape[1]):
            plt.scatter(true_points[:, i], np.zeros(len(true_points)) + i * 0.05, marker="x")
    ax.legend()
    ax.set_title(name)
    ax.set_xlabel("Factor Scores")
    ax.set_ylim([-0.05, 1.5])
    return fig


def make_acc_over_iteration(acc_by_factor):
    num_factors = 3
    num_iterations = 3
    labels = ["competent", "broken", "curious"]
    for i in range(num_factors):
        plt.errorbar(np.arange(0, num_iterations), acc_by_factor[i].mean(0), yerr=acc_by_factor[i].std(0) / 2,
                     label=labels[i])
    plt.gca().set_xlabel("Iteration")
    plt.gca().set_ylabel("Accuracy")
    plt.legend()
    plt.gca().set_ylim((.2, .6))
    plt.gca().set_xticks(np.arange(0, 3))
    return plt.gcf()
