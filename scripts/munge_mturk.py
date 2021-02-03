import glob
import json
import os

import pandas as pd
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer

import numpy as np
from joblib import load, dump

from matplotlib.backends.backend_pdf import PdfPages

import krippendorff

from models.plotting import plot_histograms, make_scatterplot, make_fa_plots, make_conf_mat_plots, make_density
from models.simple import fit_lin_reg, fit_svm, bin_factor_score, bin_likert, fit_mlp, fit_mlp_regressor, fit_log_reg, \
    fit_knn
from models.util import process_turk_files, question_names, feature_names

factor_structure = {"competent":
                        ["competent", "efficient", "energetic", "focused", "intelligent", "reliable", "responsible"],
                    "broken": ["broken", "clumsy", "confused", "lazy", "lost"],
                    "curious": ["curious", "investigative"]
                    }

condition_ratings = None
demos = None
other_data = None

demo_trajs = []
demo_exps = []
demo_prompts = []
demo_wids = []
for base in ["pilot1","active1","active2"]:
    cr, d, o = process_turk_files(base + ".csv", traj_file=base + "_trajs.json")
    # Fill in missing values
    cr[cr[question_names] == 6] = np.nan
    imp = IterativeImputer(missing_values=np.nan, max_iter=200, random_state=0, min_value=1, max_value=5)
    to_impute = cr[question_names].to_numpy()
    cr[question_names] = np.rint(imp.fit_transform(to_impute)).astype(int)
    assert not cr[cr[question_names] == np.nan].any().any()
    assert not cr[cr[question_names] == 6].any().any()
    if condition_ratings is not None:
        condition_ratings = pd.concat([condition_ratings, cr])
        demos = pd.concat([demos, d])
        other_data = pd.concat([other_data, o])
    else:
        condition_ratings = cr
        demos = d
        other_data = o

    demo_trajs += list(d["traj"])
    demo_exps += list(d["explain_traj"])
    demo_prompts += list(d["aid"])
    demo_wids += list(d["WorkerId"])

data = {"trajectories": str(demo_trajs), "explanations": demo_exps, "prompts": demo_prompts, "workerIds": demo_wids}
with open("all_demos.json", 'w') as f:
    json.dump(data, f)

# Create factor loadings
num_factors = 3
exp_transformer = FactorAnalyzer(num_factors)
analysis = exp_transformer.fit(condition_ratings[question_names].to_numpy())
print("Eignvalues:", analysis.get_eigenvalues())
# Use this line to keep the model from the pilot
# dump(exp_transformer, "factor_model.pickle")

fa_plot = make_fa_plots(condition_ratings, analysis)

factor_names = ["factor" + str(i) for i in range(num_factors)]
model_spec = ModelSpecificationParser.parse_model_specification_from_dict(condition_ratings[question_names].to_numpy(),
                                                                          factor_structure)
cfa = ConfirmatoryFactorAnalyzer(model_spec, 42)
cfa_res = cfa.fit(condition_ratings[question_names])
exp_transformer = load("factor_model.pickle")
condition_ratings[factor_names] = exp_transformer.transform(condition_ratings[question_names].to_numpy())

# factor_score_weights = pd.DataFrame(analysis.loadings_.T, columns=question_names)


fit_lin_reg(condition_ratings, question_names)


factor_names = ["factor" + str(i) for i in range(3)]


cond_qual = []
for i in range(21):
    cond_qual.append([])

by_condition = condition_ratings.groupby("id")
for name, group in by_condition:
    # print(name)
    for _, (describe, explain, wid) in group[["describe", "explain", "WorkerId"]].iterrows():
        # print(describe)
        # print(explain)
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
    # print("")

demos_by_worker = demos.groupby("WorkerId")
for worker, group in demos_by_worker:
    print(worker)
    for _, (aid, explain) in group[["aid", "explain_traj"]].iterrows():
        print("({}) - {}".format(aid, explain))
    print("")
    # print("")

con_hists = []

for condition_code, data in by_condition:
    hists = make_density("Condition " + str(condition_code), data[factor_names])
    con_hists.append(hists)

"""for condition_code, data in by_condition:
    hists = plot_histograms("Condition " + str(condition_code), question_names, data, upper_bound=6)
    con_hists.append(hists)

combined_hists = plot_histograms("All Conditions", question_names, condition_ratings, upper_bound=80)
con_hists.append(combined_hists)
"""

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


"""# Scatterplots
feat_v_attr_scatters = []
for question in question_names:
    fig = make_scatterplot(feature_names, question, condition_ratings)
    feat_v_attr_scatters.append(fig)"""

with PdfPages('../data/plots.pdf') as pp:
    pp.savefig(fa_plot)
    pp.savefig(conf_plots)
    for fig in con_hists:
        pp.savefig(fig)
    for fig in feat_v_attr_scatters:
        pp.savefig(fig)
