"""
Summarize and run basic analysis on MTurk returns
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from joblib import load
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from models.plotting import make_fa_plots, make_density
from processing.loading import process_turk_files
from processing.mappings import short_question_names, question_names, factor_names

factor_structure = {"competent":
                        ["competent", "efficient", "focused", "intelligent", "reliable", "responsible"],
                    "broken": ["broken", "clumsy", "confused", "lost"],
                    "curious": ["curious", "investigative"]
                    }

condition_ratings = None
demos = None
other_data = None

demo_trajs = []
demo_exps = []
demo_prompts = []
demo_wids = []
for base in ["pilot1", "active1", "active2", "mdn_active1", "mdn_active2"]:
    cr, d, o, comparison = process_turk_files(base + ".csv", traj_file=base + "_trajs.json")
    q_names = [q_name for q_name in question_names if q_name in cr.columns]
    # Fill in missing values
    cr[cr[q_names] == 6] = np.nan
    imp = IterativeImputer(missing_values=np.nan, max_iter=200, random_state=0, min_value=1, max_value=5)
    to_impute = cr[q_names].to_numpy()
    cr[q_names] = np.rint(imp.fit_transform(to_impute)).astype(int)
    assert not cr[cr[q_names] == np.nan].any().any()
    assert not cr[cr[q_names] == 6].any().any()
    if condition_ratings is not None:
        condition_ratings = pd.concat([condition_ratings, cr], ignore_index=True)
        demos = pd.concat([demos, d])
        other_data = pd.concat([other_data, o])
    else:
        condition_ratings = cr
        demos = d
        other_data = o
    data = {"trajectories": str(d["traj"]), "explanations": list(d["explain_traj"]), "prompts": list(d["aid"]),
                "workerIds": list(d["WorkerId"])}

    with open(f"{base}_demos.json", 'w') as f:
        json.dump(data, f)

    demo_trajs += list(d["traj"])
    demo_exps += list(d["explain_traj"])
    demo_prompts += list(d["aid"])
    demo_wids += list(d["WorkerId"])

data = {"trajectories": str(demo_trajs), "explanations": demo_exps, "prompts": demo_prompts, "workerIds": demo_wids}
with open("all_demos.json", 'w') as f:
    json.dump(data, f)


workers = other_data.groupby("WorkerId").first()
genders = workers["Answer.gender"]
print((genders.str.slice(0, 1) == "m").sum(),
      (genders.str.slice(0, 1) == "f").sum())
print(genders[~genders.str.contains("ale")].to_string())
print(len(workers), workers["Answer.age"].min(), workers["Answer.age"].max(), workers["Answer.age"].mean(), workers["Answer.age"].std())

# Create factor loadings
# See the factor analysis script for more exploration steps
num_factors = 3
exp_transformer = FactorAnalyzer(num_factors)
analysis = exp_transformer.fit(condition_ratings[short_question_names].to_numpy())
print("Eignvalues:", analysis.get_eigenvalues())
# Use this line to keep the model from the pilot
# dump(exp_transformer, "factor_model.pickle")

fa_plot = make_fa_plots(condition_ratings, analysis, short_question_names)

model_spec = ModelSpecificationParser.parse_model_specification_from_dict(condition_ratings[short_question_names].to_numpy(),
                                                                          factor_structure)
cfa = ConfirmatoryFactorAnalyzer(model_spec, max_iter=20000)
questions_in_factor_order = [value for factor in factor_structure.values() for value in factor]
cfa_res = cfa.fit(condition_ratings[questions_in_factor_order])
exp_transformer = load("factor_model.pickle")
condition_ratings[factor_names] = exp_transformer.transform(condition_ratings[short_question_names].to_numpy())

# factor_score_weights = pd.DataFrame(analysis.loadings_.T, columns=question_names)


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

con_density_plots = []

# Can be helpful to see an estimate of the distribution (via KDE)
# But note that you may have very few samples per condition
for condition_code, data in by_condition:
    densities = make_density("Condition " + str(condition_code), data[factor_names])
    con_density_plots.append(densities)

"""for condition_code, data in by_condition:
    hists = plot_histograms("Condition " + str(condition_code), question_names, data, upper_bound=6)
    con_hists.append(hists)

combined_hists = plot_histograms("All Conditions", question_names, condition_ratings, upper_bound=80)
con_hists.append(combined_hists)
"""


"""# Scatterplots
feat_v_attr_scatters = []
for question in question_names:
    fig = make_scatterplot(feature_names, question, condition_ratings)
    feat_v_attr_scatters.append(fig)"""

fig = sns.pairplot(condition_ratings[factor_names])
fig.set(xlim=(-3, 3))
fig.set(ylim=(-3, 3))
plt.show()


with PdfPages('../data/plots.pdf') as pp:
    pp.savefig(fa_plot)

    for fig in con_density_plots:
        pp.savefig(fig)

