import json
import os
from functools import partial

import pandas as pd
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer

import numpy as np
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from models.mdn.mdn import marginal_mog_log_prob
from models.mdn.ensemble import MDNEnsemble
from processing.loading import process_turk_files
from processing.mappings import short_question_names, question_names, factor_structure, eval_2_cond_names
from search.coverage import trajectory_features
from search.util import load_map
import pingouin as pg

factor_names = ["f0:competence", "f1:brokenness", "f2:curiosity"]


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
    dists = pd.DataFrame(index=by_condition.first().index, columns=["score", "density"])
    cond_batch = torch.Tensor(np.vstack(by_condition.first()["features"]))
    n_points = 300
    x = np.linspace(-3, 3, n_points)
    x_b = torch.Tensor(x)
    marg_log_prob = marginal_mog_log_prob(*ens.forward(cond_batch), x_b.reshape([-1, 1, 1]))
    mean_marg_prob = torch.mean(torch.exp(marg_log_prob), dim=1)
    mean_marg_prob = mean_marg_prob.detach().numpy().reshape(12, n_points)
    cols = [f + "0" for f in factor_names] + [f + "1" for f in factor_names] + [f + "2" for f in factor_names] + [
        f + "3" for f in factor_names]

    # On why we can't do straight LLR test https://stats.stackexchange.com/questions/137557/comparison-of-log-likelihood-of-two-non-nested-models
    model_density = pd.melt(pd.DataFrame(mean_marg_prob.T, index=x, columns=cols).reset_index(), ["index"],
                            value_name="density", var_name="condition")
    model_density["factor"] = model_density["condition"].str.slice(0, -1)
    model_density["condition"] = model_density["condition"].str.slice(-1).astype(int)
    model_density["condition"] = model_density["condition"].map(lambda x: by_condition.first().index[x])
    model_density.rename({"index": "factor_score"}, axis=1, inplace=True)

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
    return condition_ratings, comparison, model_density


exp_names = ["in_competence", "in_curiosity", "in_brokenness", "test_competence", "test_curiosity", "test_brokenness"]
for exp_name in exp_names:
    condition_ratings, comparison, model_density = analyze_experiment(exp_name)

    focus_factor = exp_name.split("_")[1]
    for factor_name in factor_names:
        if focus_factor not in factor_name:
            continue
            condition_ratings.drop(columns=[factor_name],axis=1, inplace=True)
        else:
            focus_factor = factor_name
    plt.rcParams["font.family"] = "Times New Roman"
    melted_choice = comparison.melt(value_vars=["most_id", "least_id"], id_vars=["experiment"], var_name="extreme",
                                    value_name="condition")
    choice = sns.catplot(data=melted_choice, x="condition", order=["1x", "2x", "4x", "12x"], row="extreme",
                         col="experiment", kind="count", height=1.5, aspect=2.0 / 2.0)

    #melted_factors = condition_ratings.melt(id_vars="id", value_vars=["f0:competence", "f1:brokenness", "f2:curiosity"],
    #                                       var_name="factor", value_name="factor_score")
    # ax = sns.swarmplot(data=condition_ratings, x="f0:competence", y="id")
    swarm = sns.catplot(data=condition_ratings, x=focus_factor, row="id", kind="swarm",
                        row_order=["1x", "2x", "4x", "12x"])
    swarm.set(xlim=(-3, 3))

    fig, axs = plt.subplots(ncols=1, nrows=4, sharex = True, figsize=(2.5, 5))
    for i, condition in enumerate(["1x", "2x", "4x", "12x"]):
        ax = axs[i]
        focus_ratings = condition_ratings[condition_ratings["id"] == condition]
        ax.set_xlim((-3,3))
        sns.swarmplot(data=focus_ratings, x=focus_factor, ax=axs[i], size=3, color=sns.color_palette()[i])
        ax.set_ylim(reversed(ax.get_ylim()))
        focus_density = model_density[model_density["factor"] == focus_factor]
        focus_density_condition = focus_density[focus_density["condition"] == condition]
        max_density = focus_density_condition["density"].max()
        ax.set_ylim((-.5, max_density + .05))
        sns.lineplot(data=focus_density_condition, x="factor_score", y="density", ax=axs[i], color="grey")
        ax.set_xlabel(None)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        #ax.set_yticks([])
        ax.set_ylabel(condition)

    #plt.show()
    #sns.despine(fig=fig, bottom=True, left=True)
    with PdfPages(f'../data/material/plots_{exp_name}.pdf') as pp:
        pp.savefig(fig)
        #pp.savefig(pred.fig)
        pp.savefig(choice.fig)
