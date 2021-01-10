import glob
import json

import pandas as pd
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer

import numpy as np
from joblib import load, dump

from matplotlib.backends.backend_pdf import PdfPages

import krippendorff
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from models.plotting import plot_histograms, make_scatterplot, make_fa_plots, make_conf_mat_plots
from models.simple import fit_lin_reg, fit_classification, bin_factor_score, bin_likert
from models.util import process_turk_files, question_names, feature_names, load_trajectory_pool

factor_structure = {"competent":
                        ["competent", "efficient", "energetic", "focused", "intelligent", "reliable", "responsible"],
                    "broken": ["broken", "clumsy", "confused", "lazy", "lost"],
                   "curious": ["curious", "investigative"]
                    }

condition_ratings, demos, other_data = process_turk_files(["active1.csv"])
trajectories, features = load_trajectory_pool("active1.json")

demo_trajs = list(demos["traj"])
demo_exps = list(demos["explain_traj"])
demo_prompts = list(demos["aid"])
demo_wids = list(demos["WorkerId"])

data = {"trajectories": str(demo_trajs), "explanations": demo_exps, "prompts": demo_prompts, "workerIds": demo_wids}
with open("active1_demos.json", 'w') as f:
    json.dump(data, f)

# Add feats and traj data
condition_ratings["features"] = condition_ratings["id"].apply(lambda x: features[x]).to_numpy()
condition_ratings["trajectories"] = condition_ratings["id"].apply(lambda x: trajectories[x])

# Fill in missing values
imp = IterativeImputer(missing_values=6, max_iter=20, random_state=0)
to_impute = condition_ratings[question_names].to_numpy()
imp.fit(to_impute)
condition_ratings[question_names] = np.rint(imp.transform(to_impute)).astype(int)

# Create factor loadings
num_factors = 3
exp_transformer = FactorAnalyzer(num_factors)
analysis = exp_transformer.fit(condition_ratings[question_names].to_numpy())
print("Eignvalues:", analysis.get_eigenvalues())
# Use this line to keep the model from the pilot
# dump(exp_transformer, "factor_model.pickle")

fa_plot = make_fa_plots(condition_ratings, analysis)


model_spec = ModelSpecificationParser.parse_model_specification_from_dict(condition_ratings[question_names].to_numpy(), factor_structure)
cfa = ConfirmatoryFactorAnalyzer(model_spec,42)
cfa_res = cfa.fit(condition_ratings[question_names])
exp_transformer = load("factor_model.pickle")
transformed = exp_transformer.transform(condition_ratings[question_names].to_numpy())

factor_score_weights = pd.DataFrame(analysis.loadings_.T, columns=question_names)
factor_score_weights[factor_score_weights > .4] = 1
factor_score_weights[factor_score_weights <= .4] = 0

factor_names = ["factor" + str(i) for i in range(num_factors)]
condition_ratings[factor_names] = transformed
for i in range(num_factors):
    num_components = factor_score_weights.iloc[i].sum()
    condition_ratings["afactor"+str(i)] = (condition_ratings[question_names] * factor_score_weights.iloc[i]).sum(axis=1) / num_components

fit_lin_reg(condition_ratings, question_names)

factor_names = ["factor" + str(i) for i in range(3)]

x = np.vstack(condition_ratings["features"])
y = condition_ratings[question_names + factor_names].copy()
x, x_test, y, y_test = train_test_split(x, y, test_size=.2, random_state=0)

y[question_names] = bin_likert(y[question_names].copy())
y_test[question_names] = bin_likert(y_test[question_names].copy())

y[factor_names] = bin_factor_score(y[factor_names].copy())
y_test[factor_names] = bin_factor_score(y_test[factor_names].copy())
"""
print("Classification over features targeting items---------------")
for i, attr_name in enumerate(question_names):
    clf = fit_classification(x, y[attr_name].to_numpy())
    print("svm acc {} \t\t {:.2}".format(attr_name, clf.score(x_test, y_test[attr_name].to_numpy())))
"""
print("Classification over features targeting factors---------------")

factor_clfs = []
conf_mats = []

for i, factor_name in enumerate(factor_names):
    clf = fit_classification(x, y[factor_name])
    factor_clfs.append(clf)
    y_test.reset_index(None, drop=True, inplace=True)
    print("svm acc {} \t\t {:.2}".format(factor_name, clf.score(x_test, y_test[factor_name].to_numpy())))
    conf_mats.append(confusion_matrix(y_test[factor_name], clf.predict(x_test), sample_weight=None, normalize=None))

dump(factor_clfs, "factor_svms.pickle")
conf_plots = make_conf_mat_plots(factor_clfs, x_test, [y_test[f] for f in factor_names])

cond_qual = []
for i in range(21):
    cond_qual.append([])

by_condition = condition_ratings.groupby("id")
for name, group in by_condition:
    #print(name)
    for _, (describe, explain, wid) in group[["describe", "explain", "WorkerId"]].iterrows():
        #print(describe)
        #print(explain)
       # print("")
        cond_qual[int(name)].append(describe + "---" + explain + "--" + wid)

data = {"summary": cond_qual}
with open("batch_qual.json", 'w') as f:
    json.dump(data, f)

turker_performance = pd.DataFrame()
turker_performance["HITTime"] = other_data.groupby("WorkerId")["WorkTimeInSeconds"].mean()
turker_performance["Comment"] = other_data.groupby("WorkerId")["Answer.comment"].apply(list)
turker_performance.to_csv("turker_stats.txt", index=True)

# By worker
by_worker = condition_ratings.groupby("WorkerId")
for worker, group in by_worker:
    print(worker)
    for _, (describe, explain, wid) in group[["describe", "explain", "WorkerId"]].iterrows():
        print(describe)
        print(explain)
    print("")
       #print("")

demos_by_worker = demos.groupby("WorkerId")
for worker, group in demos_by_worker:
    print(worker)
    for _, (aid, explain) in group[["aid","explain_traj"]].iterrows():
        print("({}) - {}".format(aid, explain))
    print("")
       #print("")



con_hists = []
for condition_code, data in by_condition:
    hists = plot_histograms("Condition " + str(condition_code), question_names, data, upper_bound=6)
    con_hists.append(hists)

combined_hists = plot_histograms("All Conditions", question_names, condition_ratings, upper_bound=80)
con_hists.append(combined_hists)


def calculate_alphas():
    alphas = []
    for i in range(num_factors):
        factor_by_worker = []
        for worker_id, worked in by_worker:
            factor_by_worker.append([])
            factor_by_worker[-1] = []
            for _ in range(21):
                factor_by_worker[-1].append(np.nan)
            for _, row in worked.iterrows():
                for factor in range(num_factors):
                    factor_by_worker[-1][row["id"]] = row["factor" + str(i)]
        alphas.append(krippendorff.krippendorff.alpha(factor_by_worker))

    print("Krippendorf alphas")
    print(alphas)

# Scatterplots
feat_v_attr_scatters = []
for question in question_names:
    fig = make_scatterplot(feature_names, question, condition_ratings)
    feat_v_attr_scatters.append(fig)

with PdfPages('../data/plots.pdf') as pp:
    pp.savefig(fa_plot)
    pp.savefig(conf_plots)
    for fig in con_hists:
        pp.savefig(fig)
    for fig in feat_v_attr_scatters:
        pp.savefig(fig)
