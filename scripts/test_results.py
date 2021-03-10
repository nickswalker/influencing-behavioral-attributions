import glob
import itertools
import json
import os
from functools import partial

import pandas as pd
import torch
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer

import numpy as np
from joblib import load, dump
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from models.mdn import marginal_mog_log_prob
from models.nn import load_ensemble
from models.plotting import plot_histograms, make_scatterplot, make_fa_plots, make_conf_mat_plots, make_density
from models.simple import fit_lin_reg
from models.util import process_turk_files, question_names, feature_names, old_question_names, short_question_names
from search.coverage import trajectory_features
from search.util import load_map
import pingouin as pg

cond_names = {0:"TSK", 1: "B+",2:"B-",3:"BAL", 4:"TSK", 5: "B+",6:"B-",7:"BAL", 8: "B+",9:"B-",10:"BAL"}
test_cond_names = {0:"TSK", 1: "B+",2:"B-",3:"BAL", 4: "B+",5:"B-",6:"BAL", 7: "B+",8:"B-",9:"BAL"}
factor_structure = {"competent":
                        ["competent", "efficient", "focused", "intelligent", "reliable", "responsible"],
                    "broken": ["broken", "clumsy", "confused", "lost"],
                    "curious": ["curious", "investigative"]
                    }

condition_ratings = None
comparison = None
other_data = None

exp_name = "test_competent"
if "test" in exp_name:
    cond_names = test_cond_names
import sys

sys.stdout = open(f"all{exp_name}_data.txt", 'w')

for base in ["test_competent", "test_broken", "test_curious"]:
    cr, _, o, comp = process_turk_files(base + ".csv", traj_file=base + "_trajs.json")
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
condition_ratings["num_id"] =  condition_ratings["id"]
condition_ratings["id"] = condition_ratings["id"].map(cond_names)
comparison["most_id"] = comparison["most_id"].map(cond_names)
comparison["least_id"] = comparison["least_id"].map(cond_names)
workers = other_data.groupby("WorkerId").first()
genders = workers["Answer.gender"]
print(f"{(genders.str.slice(0, 1) == 'm').sum()}  male, {(genders.str.slice(0, 1) == 'f').sum()} female")
print(genders[~genders.str.contains("ale")].to_string())
print("N", len(workers), "min age", workers["Answer.age"].min(), "max age", workers["Answer.age"].max(), "M", workers["Answer.age"].mean(),"SD", workers["Answer.age"].std())

alpha = []
for factor_name, components in factor_structure.items():
    alpha.append(pg.cronbach_alpha(condition_ratings[components])[0])
print("Cronbach's alpha by factor:", alpha)

# Create factor loadings
num_factors = 3
factor_names = ["f0:competence", "f1:brokenness", "f2:curiosity"]

exp_transformer = load("factor_model.pickle")
condition_ratings[factor_names] = exp_transformer.transform(condition_ratings[short_question_names].to_numpy())

grid, bedroom = load_map("../interface/assets/map32.tmx")

featurizer = partial(trajectory_features, bedroom, grid)
condition_ratings["features"] = condition_ratings["trajectories"].apply(lambda x: featurizer(x))
ens = load_ensemble(os.getcwd() + "/final")


cond_qual = {}
for cond in cond_names.values():
    cond_qual[cond] = []

out = -torch.log(ens.mean_prob(torch.Tensor(np.vstack(condition_ratings["features"])), torch.Tensor(condition_ratings[factor_names].to_numpy())))
condition_ratings["logprob"] = out.detach().numpy()

breakdown = condition_ratings.melt(id_vars=["id","experiment"], value_vars="logprob",value_name="logprob").drop(columns=["variable"]).groupby(["id","experiment"])["logprob"].describe()

by_condition = condition_ratings.groupby("id")
print("AVG NLL\n", by_condition.describe()["logprob"][["mean", "std"]])
print("AVG NRL OVERALL\n", condition_ratings["logprob"].mean())
dists = pd.DataFrame(index=by_condition.first().index, columns=["score", "density"])
cond_batch = torch.Tensor(np.vstack(by_condition.first()["features"]))
n_points = 30
x = np.linspace(-3, 3, n_points)
x_b = torch.Tensor(x)
marg_log_prob = marginal_mog_log_prob(*ens.forward(cond_batch), x_b.reshape([-1, 1, 1]))
mean_marg_prob = torch.mean(torch.exp(marg_log_prob),dim=1)
mean_marg_prob = mean_marg_prob.detach().numpy().reshape(12, n_points)
cols = [f+"0" for f in factor_names] + [f+"1" for f in factor_names] + [f + "2" for f in  factor_names] + [f +"3" for f in factor_names]
melted = pd.melt(pd.DataFrame(mean_marg_prob.T, index=x, columns=cols).reset_index(), ["index"], value_name="density", var_name="condition")
melted["factor"] = melted["condition"].str.slice(0,-1)
melted["condition"] = melted["condition"].str.slice(-1).astype(int)
melted["condition"] = melted["condition"].map(lambda x: by_condition.first().index[x])
melted.rename({"index": "factor_score"}, axis=1,inplace=True)
for name, group in by_condition:
    print("------------")

    print(name)
    for _, (describe, explain, wid) in group[["describe", "explain", "WorkerId"]].iterrows():
        print(describe)
        print(explain)
        print("")
        cond_qual[name].append(describe + "---" + explain + "--" + wid)

print("**********************")
print("MOST")
by_condition = comparison.groupby("most_id")
for name, group in by_condition:
    print("------------")
    print(name)
    for explain in group["most_explain"]:
        print(explain)

print("LEAST")
by_condition = comparison.groupby("least_id")
for name, group in by_condition:
    print("------------")
    print(name)
    for explain in group["least_explain"]:
        print(explain)


data = {"summary": cond_qual}
with open("batch_qual.json", 'w') as f:
    json.dump(data, f)
"""
print("**********************")
post_results = ""
scale_value_by_condition = []
anova_results = []
for scale in factor_names:
    # To get full p values
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    anova_result = pg.rm_anova(condition_ratings, scale, subject='WorkerId', within=['id'])
    post_results += "Scale: {}\n".format(scale)
    post_results += "RM ANOVA\n"
    post_results += anova_result.to_string() + "\n\n"
    anova_results.append(anova_result)

    if anova_result["p-unc"][0] > 0.05:
        post_results += "-------------- \n\n\n"
        continue


    # Run all tests and apply post-hoc corrections
    res = pg.pairwise_ttests(condition_ratings, scale, subject='WorkerId', within=['num_id'], tail="two-sided",
                             padjust="holm", return_desc=True)
    res["A"] = res["A"].map(cond_names)
    res["B"] = res["B"].map(cond_names)


    post_results += res.to_string() + "\n"
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    post_results += res.to_latex(columns=["A","B","mean(A)","std(A)","mean(B)","std(B)","T","p-corr","hedges"],index=False)
    post_results += "--------------\n\n\n"

print(post_results)
"""

turker_performance = pd.DataFrame()
turker_performance["HITTime"] = other_data.groupby("WorkerId")["WorkTimeInSeconds"].mean()
turker_performance["Comment"] = other_data.groupby("WorkerId")["Answer.comment"].apply(list)
turker_performance.to_csv("turker_stats.txt", index=True)

print("**********************")
# By worker
by_worker = condition_ratings.groupby("WorkerId")
for worker, group in by_worker:
    print(worker)
    for _, (describe, explain, wid) in group[["describe", "explain", "WorkerId"]].iterrows():
        print(describe)
        print(explain)
    print("")
    # print("")

plt.rcParams["font.family"] = "Times New Roman"
melted_choice = comparison.melt(value_vars=["most_id", "least_id"], id_vars=["experiment"], var_name="extreme", value_name="condition")
choice = sns.catplot(data=melted_choice, x="condition",order=["TSK","B+","B-","BAL"], row="extreme", col="experiment", kind="count", height=1.5, aspect=2.0/2.0)


melted_factors = condition_ratings.melt(id_vars="id", value_vars=["f0:competence", "f1:brokenness", "f2:curiosity"], var_name="factor", value_name="factor_score")
#ax = sns.swarmplot(data=condition_ratings, x="f0:competence", y="id")
swarm = sns.catplot(data=melted_factors, x="factor_score", y="factor", row="id", hue="factor",kind="swarm", row_order=["TSK","B+","B-","BAL"])
swarm.set(xlim=(-3,3))

pred = sns.relplot(
    data=melted, x="factor_score", y="density",
    row="condition", hue="factor",
    kind="line", row_order=["TSK","B+","B-","BAL"]
)


with PdfPages(f'../data/test.pdf') as pp:
    #pp.savefig(swarm.fig)
    #pp.savefig(pred.fig)
    pp.savefig(choice.fig)

