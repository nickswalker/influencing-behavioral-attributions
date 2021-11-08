import json
import os
from functools import partial

import pandas as pd
import scipy.stats
import statsmodels.stats.multitest
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer
from sklearn.neighbors import KernelDensity

import numpy as np
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from models.mdn.mdn import marginal_mog_log_prob, marginal_mog, mog_log_prob
from models.mdn.ensemble import MDNEnsemble
from processing.loading import process_turk_files
from processing.mappings import short_question_names, question_names, factor_structure, eval_2_cond_names
from search.coverage import trajectory_features
from search.util import load_map
import pingouin as pg

from mpl_toolkits.axes_grid1 import make_axes_locatable
from processing.mappings import  factor_names


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def ks_test(ens, observed_by_condition, factor_names):
    results = []
    for cond, data in observed_by_condition:
        print(cond)
        cond_feats = data.iloc[0]["features"]
        def make_callable_cdf(ens, d):
            cond_feats_t = torch.Tensor(cond_feats).reshape([1, -1])
            mog_params = ens.forward(cond_feats_t)
            def callable_cdf(eval_x):
                n_points = 600
                x = np.linspace(-3, 3, n_points)
                x_t = torch.Tensor(x)
                mog_d = marginal_mog(mog_params, d)
                log_probs = mog_log_prob(*mog_d, x_t.reshape([-1,1,1]))
                mean_marg_prob = torch.mean(torch.exp(log_probs), dim=1)
                up_to = x.reshape(-1, 1) < eval_x
                masked = mean_marg_prob.repeat((len(eval_x), 1)).t()
                masked[~up_to]  = 0
                cdf_at_x = torch.trapz(masked.t(), x_t.reshape([1,-1]))
                cdf_at_x = cdf_at_x.detach().numpy()
                return cdf_at_x
            return callable_cdf
        for i, factor in enumerate(factor_names):
            model_cdf = make_callable_cdf(ens, i)
            obs = data[factor]
            res = scipy.stats.ks_1samp(obs, model_cdf)
            results.append((cond, factor, res.statistic, res.pvalue))
    return pd.DataFrame(data=results, columns=["condition", "factor", "statistic", "p"])


def kendalls_test(observed, factor_names):
    results = []
    for factor in factor_names:
        x = observed["id"].str.slice(0,-1).astype(int).to_numpy()
        y = observed[factor].to_numpy()
        sort = np.lexsort((y,x))
        x = x[sort]
        y = y[sort]
        res = scipy.stats.kendalltau(x, y)
        results.append((factor, res.correlation, res.pvalue))
    return pd.DataFrame(data=results, columns=["factor", "correlation", "p"])


def pairwise_test(data, condition_names):
    post_results = ""
    scale_value_by_condition = []
    anova_results = []
    for scale in factor_names:
        # To get full p values
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        anova_result = pg.rm_anova(data, scale, subject='WorkerId', within=['id'])
        post_results += "Scale: {}\n".format(scale)
        post_results += "RM ANOVA\n"
        post_results += anova_result.to_string() + "\n\n"
        anova_results.append(anova_result)

        if anova_result["p-unc"][0] > 0.05:
            post_results += "-------------- \n\n\n"
            continue

        # Run all tests and apply post-hoc corrections
        res = pg.pairwise_ttests(data, scale, subject='WorkerId', within=['num_id'], tail="two-sided",
                                 padjust="holm", return_desc=True)
        res["A"] = res["A"].map(condition_names)
        res["B"] = res["B"].map(condition_names)

        post_results += res.to_string() + "\n"
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        post_results += res.to_latex(
            columns=["A", "B", "mean(A)", "std(A)", "mean(B)", "std(B)", "T", "p-corr", "hedges"], index=False)
        post_results += "--------------\n\n\n"
    return post_results


def compose_qualitative(ratings, comparison, condition_names):
    out = ""
    cond_qual = {}
    for cond in condition_names.values():
        cond_qual[cond] = []

    by_condition = ratings.groupby("id")
    for name, group in by_condition:
        out += "------------\n"

        out += name + "\n"
        for _, (describe, explain, wid) in group[["describe", "explain", "WorkerId"]].iterrows():
            out += describe + "\n"
            out += explain + "\n"
            out += "\n"
            cond_qual[name].append(describe + "---" + explain + "--" + wid)

    out += "**********************\n"
    out += "MOST\n"
    by_condition = comparison.groupby("most_id")
    for name, group in by_condition:
        out += "------------\n"
        out += name + "\n"
        for explain in group["most_explain"]:
            out += explain + "\n"

    out += "LEAST\n"
    by_condition = comparison.groupby("least_id")
    for name, group in by_condition:
        out += "------------\n"
        out += name + "\n"
        for explain in group["least_explain"]:
            out += explain + "\n"

    out += "**********************\n"
    # By worker
    by_worker = ratings.groupby("WorkerId")
    for worker, group in by_worker:
        out += worker + "\n"
        for _, (describe, explain, wid) in group[["describe", "explain", "WorkerId"]].iterrows():
            out += describe + "\n"
            out += explain + "\n"
        out += "\n"
    return out, cond_qual


def analyze_experiment(exp_name):
    traj_file = "in.json"
    env_name = "house.tmx"
    condition_names = eval_2_cond_names
    if "test" in exp_name:
        traj_file = "test.json"
        env_name = "house_test.tmx"
    import sys
    grid, bedroom = load_map(f"../interface/assets/{env_name}")

    featurizer = partial(trajectory_features, bedroom, grid)

    old_stdout = sys.stdout
    sys.stdout = open(f"material/{exp_name}_data.txt", 'w')
    condition_ratings = None
    comparison = None
    other_data = None
    for base in [exp_name]:
        cr, _, o, comp = process_turk_files(base + ".csv", traj_file=traj_file, featurizer=featurizer)
        cr["experiment"] = base
        q_names = [q_name for q_name in question_names if q_name in cr.columns]
        # Fill in missing values
        cr[cr[q_names] == 6] = np.nan
        imp = IterativeImputer(missing_values=np.nan, max_iter=200, random_state=0, min_value=1, max_value=5)
        to_impute = cr[q_names].to_numpy()
        cr[q_names] = np.rint(imp.fit_transform(to_impute)).astype(int)
        assert not cr[cr[q_names] == np.nan].any().any()
        assert not cr[cr[q_names] == 6].any().any()
        comp["experiment"] = base
        if condition_ratings is not None:
            condition_ratings = pd.concat([condition_ratings, cr])
            comparison = pd.concat([comparison, comp])
            other_data = pd.concat([other_data, o])
        else:
            condition_ratings = cr
            comparison = comp
            other_data = o
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    condition_ratings["num_id"] = condition_ratings["id"]
    condition_ratings["id"] = condition_ratings["id"].map(condition_names)
    comparison["most_id"] = comparison["most_id"].map(condition_names)
    comparison["least_id"] = comparison["least_id"].map(condition_names)
    workers = other_data.groupby("WorkerId").first()
    genders = workers["Answer.gender"]
    print(f"{(genders.str.slice(0, 1) == 'm').sum()}  male, {(genders.str.slice(0, 1) == 'f').sum()} female")
    print(genders[~genders.str.contains("ale")].to_string())
    print("N", len(workers), "min age", workers["Answer.age"].min(), "max age", workers["Answer.age"].max(), "M",
          workers["Answer.age"].mean(), "SD", workers["Answer.age"].std())

    alpha = []
    for factor_name, components in factor_structure.items():
        alpha.append(pg.cronbach_alpha(condition_ratings[components])[0])
    print("Cronbach's alpha by factor:", alpha)

    # Create factor loadings
    exp_transformer = load("factor_model.pickle")
    condition_ratings[factor_names] = exp_transformer.transform(condition_ratings[short_question_names].to_numpy())


    condition_ratings["features"] = condition_ratings["trajectories"].apply(lambda x: featurizer(x))
    ens = MDNEnsemble.load_ensemble(os.getcwd() + "/final_new_feats")

    out = -torch.log(ens.mean_prob(torch.Tensor(np.vstack(condition_ratings["features"])),
                                   torch.Tensor(condition_ratings[factor_names].to_numpy())))
    condition_ratings["logprob"] = out.detach().numpy()

    breakdown = condition_ratings.melt(id_vars=["id", "experiment"], value_vars="logprob", value_name="logprob").drop(
        columns=["variable"]).groupby(["id", "experiment"])["logprob"].describe()

    by_condition = condition_ratings.groupby("id")
    print("AVG NLL\n", by_condition.describe()["logprob"][["mean", "std"]])
    print("AVG NRL OVERALL\n", condition_ratings["logprob"].describe()[["mean", "std"]])
    cond_names = by_condition.first().index
    cond_batch = torch.Tensor(np.vstack(by_condition.first()["features"]))
    n_points = 600
    x = np.linspace(-3, 3, n_points)
    x_b = torch.Tensor(x)
    marg_log_prob = marginal_mog_log_prob(*ens.forward(cond_batch), x_b.reshape([-1, 1, 1]))
    mean_marg_prob = torch.mean(torch.exp(marg_log_prob), dim=1)
    mean_marg_prob = mean_marg_prob.detach().numpy()
    # On why we can't do straight LLR test https://stats.stackexchange.com/questions/137557/comparison-of-log-likelihood-of-two-non-nested-models
    new_data = []
    for i, cond_name in enumerate(cond_names):
        for j, factor in enumerate(factor_names):
            new_data.append((cond_name, factor, mean_marg_prob[i, j]))
    model_density = pd.DataFrame(data=new_data, columns=["condition", "factor", "density"])

    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    print("**********************")
    # This is not an equivalence test, but rather a test of difference
    # two one-sided tests have been suggested as an equivalence testing procedure
    # https://stats.stackexchange.com/questions/97556/is-there-a-simple-equivalence-test-version-of-the-kolmogorov-smirnov-test?rq=1
    # https://stats.stackexchange.com/questions/174024/can-you-use-the-kolmogorov-smirnov-test-to-directly-test-for-equivalence-of-two?rq=1
    ks_results = ks_test(ens, by_condition, factor_names)
    for group, data in ks_results.groupby("factor"):
        print(group)
        print(data.reset_index())
    #pvals = [0.039883, 0.001205, 0.310183, 0.043085, 0.179424, 0.026431, 0.344007, 0.127182, 0.267323, 0.125909, 0.837506, 0.652114]
    #adj = statsmodels.stats.multitest.multipletests(pvals)
    #pvals = [0.091473, 0.005065, 0.015585, 0.360311, 0.205270, 0.089199, 0.594448, 0.071204, 0.685286, 0.013982, 0.025368, 0.085334]
    #adj2 = statsmodels.stats.multitest.multipletests(pvals)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print("*************************")


    print("**********************")
    post_results = pairwise_test(condition_ratings, condition_names)
    print(post_results)

    out, cond_qual = compose_qualitative(condition_ratings, comparison, condition_names)
    print(out)

    data = {"summary": cond_qual}
    with open(f"material/{exp_name}_qual.json", 'w') as f:
        json.dump(data, f)

    turker_performance = pd.DataFrame()
    turker_performance["HITTime"] = other_data.groupby("WorkerId")["WorkTimeInSeconds"].mean()
    turker_performance["Comment"] = other_data.groupby("WorkerId")["Answer.comment"].apply(list)
    # turker_performance.to_csv("turker_stats.txt", index=True)

    sys.stdout = old_stdout
    return condition_ratings, comparison, model_density, ks_results, other_data


exp_names = ["in_competence", "in_brokenness", "in_curiosity"]
test_names = ["test_competence", "test_brokenness",
             "test_curiosity"]

n_points = 600
x = np.linspace(-3, 3, n_points)

plt.rcParams["font.family"] = "Times New Roman"
SMALL_SIZE = 7
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
fig, axs = plt.subplots(ncols=6, nrows=4, sharex=True, figsize=(5.5, 2.5))
plt.rcParams["font.family"] = "Times New Roman"


all_dat = []

ind = np.arange(4)  # the x locations for the groups
width = 1       # the width of the bars
fig_ch, axs_ch = plt.subplots(ncols=6, nrows=1, sharex=True, sharey=True, figsize=(5.5, .75))
colors = plt.get_cmap("Dark2").colors
for exp_num, names in enumerate([exp_names, test_names]):
    print(str(exp_num) + "------------------------------------------------------------")
    all_log_prob = []
    all_cond_ratings = []
    all_ks_results = []
    for dim_i, dim in enumerate(names):
        condition_ratings, comparison, model_density, ks_results ,other_data = analyze_experiment(dim)
        all_cond_ratings.append(condition_ratings)
        all_dat.append(other_data)
        ks_results["experiment"] = dim
        all_ks_results.append(ks_results)
        all_log_prob.append(condition_ratings[["experiment", "id", "logprob"]])
        focus_factor = dim.split("_")[1]
        for factor_name in factor_names:
            if focus_factor not in factor_name:
                continue
                condition_ratings.drop(columns=[factor_name], axis=1, inplace=True)
            else:
                focus_factor = factor_name

        most_counts = comparison["most_id"].value_counts()
        least_counts = comparison["least_id"].value_counts()
        for name in ["1x", "2x", "4x", "12x"]:
            if name not in most_counts:
                most_counts[name] = 0
            if name not in least_counts:
                least_counts[name] = 0

        ax = axs_ch[dim_i + 3 * exp_num]
        ax.set_title(focus_factor.capitalize())
        ax.bar(ind - width * 1. / 5, least_counts[["1x", "2x", "4x", "12x"]], color=colors[3], width=width * 2. / 5., label="Least")
        ax.bar(ind + width * 1. / 5, most_counts[["1x", "2x", "4x", "12x"]], color=colors[4], width=width * 2. / 5., label="Most")
        ax.set_xticks(ind)
        ax.set_yticks([0, 10, 20])
        ax.set_xticklabels(["1x", "2x", "4x", "12x"])
        if exp_num == 1 and dim_i == 2:
            ax.legend(prop={'size': 4})
        for i, condition in enumerate(["1x", "2x", "4x", "12x"]):
            ax = axs[i][dim_i + 3 * exp_num]

            focus_density = model_density[model_density["factor"] == focus_factor]
            focus_density_condition = focus_density[focus_density["condition"] == condition]["density"].to_numpy()[0]
            ax.plot(x, focus_density_condition, color="grey", zorder=3)


            focus_ratings = condition_ratings[condition_ratings["id"] == condition]
            just_scores = focus_ratings[focus_factor].to_numpy().reshape(-1, 1)
            ax.set_xlim((-3, 3))
            bandwidths = np.linspace(0.01, 1, 100)
            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                {'bandwidth': bandwidths},
                                cv=LeaveOneOut())
            grid.fit(just_scores)
            kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_params_["bandwidth"]).fit(just_scores)
            kde_dens = np.exp(kde.score_samples(x.reshape(-1, 1)))
            points = np.array([x, kde_dens]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            kl_curve = np.log(kde_dens / focus_density_condition)
            kl_curve = np.clip(kl_curve, -1, 1)
            #norm = plt.Normalize(ratio.min(), ratio.max())
            norm = plt.Normalize(-1, 1)
            #lc = LineCollection(segments, cmap='PiYG', norm=norm)
            lc = LineCollection(segments, cmap='Spectral', norm=norm, zorder=2)
            # Set the values used for colormapping
            lc.set_array(kl_curve)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            ax.plot(x, scipy.stats.norm.pdf(x, loc=1.5, scale=0.3), "--", lw=1, color="lightgray", zorder=1)
            #ax.plot(x, log_dens, color="blue")
            max_density = focus_density_condition.max()
            max_kde = kde_dens.max()
            ax.set_ylim((-.05, max(max_density, max_kde) + .05))
            ax.set_xlabel(None)
            ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_yticks([])
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            #plt.colorbar(lc, ax=cax)
            if i == 0:
                ax.set_title(focus_factor.capitalize())
            if dim_i == 0 and exp_num == 0:
                ax.set_ylabel(condition, rotation=0, labelpad=8)
            else:
                ax.set_ylabel("")

    # NLL  TABLE
    all_log_prob = pd.concat(all_log_prob)
    # We're only addressing a single experiment here
    all_log_prob["experiment"] = all_log_prob["experiment"].str.replace("in_", "", regex=False,)
    all_log_prob["experiment"] = all_log_prob["experiment"].str.replace("test_", "", regex=False)
    groups = all_log_prob.groupby(["experiment", "id"])
    table = all_log_prob.groupby(["id", "experiment"]).describe()["logprob"][["mean", "std"]].unstack(1)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    fmted = table.to_latex(columns=[("mean", "competence"), ("std","competence"), ("mean", "brokenness"), ("std","brokenness"), ("mean", "curiosity"), ("std","curiosity")])
    print(fmted)
    print("By cond", all_log_prob.groupby("experiment").describe()["logprob"][["mean", "std"]].to_latex())
    print(f"ALL in {exp_num}", all_log_prob.describe()["logprob"][["mean", "std"]])
    print("------------")
    cis = []
    for (experiment, id), data in all_log_prob.groupby(["experiment", "id"]):
        series = data["logprob"].to_numpy()
        bs_mean = [np.random.choice(series, size=(24), replace=True).mean() for _ in range(1000)]
        lower_mean, upper_mean = np.percentile(bs_mean, [2.5, 97.5])
        mean_error = (upper_mean - lower_mean) / 2.0
        bs_std = [np.random.choice(series, size=(24), replace=True).std() for _ in range(1000)]
        lower_std, upper_std =  np.percentile(bs_std, [2.5, 97.5])
        std_error = (upper_std - lower_std) / 2.0
        cis.append((experiment, id, series.mean(), series.std(), lower_mean, upper_mean, mean_error, lower_std, upper_std, std_error))

    cis_data = pd.DataFrame(cis, columns=["experiment", "id", "mean", "std", "lower_mean", "upper_mean", "mean_error", "lower_std", "upper_std", "std_error"])

    cis_by_exp = []

    series = all_log_prob["logprob"].to_numpy()
    n = len(series)
    bs_mean = [np.random.choice(series, size=(n), replace=True).mean() for _ in range(1000)]
    lower_mean, upper_mean = np.percentile(bs_mean, [2.5, 97.5])
    mean_error = (upper_mean - lower_mean) / 2.0
    bs_std = [np.random.choice(series, size=(n), replace=True).std() for _ in range(1000)]
    lower_std, upper_std =  np.percentile(bs_std, [2.5, 97.5])
    std_error = (upper_std - lower_std) / 2.0
    cis_by_exp.append((series.mean(), series.std(), lower_mean, upper_mean, mean_error, lower_std, upper_std, std_error))

    cis_data_by_exp = pd.DataFrame(cis_by_exp, columns=["mean", "std", "lower_mean", "upper_mean", "mean_error", "lower_std", "upper_std", "std_error"])
    pd.set_option('display.float_format', lambda x: '%0.2f' % x)
    print(str(cis_data_by_exp["mean"].item()) + "+-" + str(cis_data_by_exp["mean_error"].item()) + "(" + str(cis_data_by_exp["std"].item()) + "+-" + str(cis_data_by_exp["std_error"].item()) + ")")

    all_cond_ratings = pd.concat(all_cond_ratings, ignore_index=True)
    all_cond_ratings["experiment"] = all_cond_ratings["experiment"].str.replace("in_", "", regex=False,)
    all_cond_ratings["experiment"] = all_cond_ratings["experiment"].str.replace("test_", "", regex=False)
    by_cond_type = all_cond_ratings.groupby("id")

    # KS TEST TABLE
    all_ks_results = pd.concat(all_ks_results)
    all_ks_results["experiment"] = all_ks_results["experiment"].str.replace("in_", "", regex=False,)
    all_ks_results["experiment"] = all_ks_results["experiment"].str.replace("test_", "", regex=False)
    only_target = all_ks_results["experiment"] == all_ks_results["factor"]
    only_target = all_ks_results[only_target]
    only_target["p-corr"] = statsmodels.stats.multitest.multipletests(only_target["p"])[1]

    ks_results = only_target.drop(columns="experiment").melt(id_vars=["condition", 'factor']).set_index(
        ["condition"]).pivot(columns=["factor", "variable"], values="value")
    pd.set_option('display.float_format', lambda x: '%0.3f' % x)
    fmted = ks_results.to_latex(columns=[("competence", "statistic"), ("competence", "p-corr"), ("brokenness", "statistic"), ("brokenness", "p-corr"), ("curiosity", "statistic"), ("curiosity","p-corr")])
    print(fmted)
    # TAU TABLE
    all_tau_res = []
    for experiment, data in all_cond_ratings.groupby(["experiment"]):
        tau_results = kendalls_test(data, [experiment])
        all_tau_res.append(tau_results)
    all_tau_res = pd.concat(all_tau_res)
    all_tau_res["p-corr"] =  statsmodels.stats.multitest.multipletests(all_tau_res["p"])[1]
    table =  all_tau_res.pivot(columns=["factor"])
    table.columns = table.columns.swaplevel(0, 1)
    pd.set_option('display.float_format', lambda x: '%0.3f' % x)
    fmted = table.to_latex(columns=[("competence", "correlation"), ("competence", "p-corr"), ("brokenness", "correlation"), ("brokenness", "p-corr"), ("curiosity", "correlation"), ("curiosity","p-corr")])
    print(fmted)

    # T tests

all_dat = pd.concat(all_dat, ignore_index=True)
workers = all_dat.groupby("WorkerId").first()
genders = workers["Answer.gender"]
print(f"{(genders.str.slice(0, 1) == 'm').sum()}  male, {(genders.str.slice(0, 1) == 'f').sum()} female")
print(genders[~genders.str.contains("ale")].to_string())
print("N", len(workers), "min age", workers["Answer.age"].min(), "max age", workers["Answer.age"].max(), "M",
      workers["Answer.age"].mean(), "SD", workers["Answer.age"].std())


with PdfPages(f'../data/material/plots_div.pdf') as pp:
    pp.savefig(fig, bbox_inches='tight', transparent="True", pad_inches=0)
with PdfPages(f'../data/material/plots_choice.pdf') as pp:
    pp.savefig(fig_ch, bbox_inches='tight', transparent="True", pad_inches=0)
