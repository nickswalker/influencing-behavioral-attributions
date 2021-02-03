import glob
import json
import os
import random

import pandas as pd
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer

import numpy as np

from sklearn.model_selection import GroupShuffleSplit

from models.nn import train_mdn
from models.simple import fit_svm, bin_factor_score, bin_likert
from models.util import process_turk_files, question_names

random.seed(0)
np.random.seed(0)

factor_structure = {"competent":
                        ["competent", "efficient", "energetic", "focused", "intelligent", "reliable", "responsible"],
                    "broken": ["broken", "clumsy", "confused", "lazy", "lost"],
                    "curious": ["curious", "investigative"]
                    }

condition_ratings = []
demos = []
other_data = []
max_id = 0
for base in ["pilot1", "active1", "active2"]:
    cr, d, o = process_turk_files(base + ".csv", traj_file=base + "_trajs.json")
    # Fill in missing values
    cr[cr[question_names] == 6] = np.nan
    imp = IterativeImputer(missing_values=np.nan, max_iter=200, random_state=0, min_value=1, max_value=5)
    to_impute = cr[question_names].to_numpy()
    cr[question_names] = np.rint(imp.fit_transform(to_impute)).astype(int)
    cr["uuid"] = cr["id"] + max_id
    max_id = cr["uuid"].max() + 1
    assert not cr[cr[question_names] == np.nan].any().any()
    assert not cr[cr[question_names] == 6].any().any()
    condition_ratings.append(cr)
    demos.append(d)
    other_data.append(o)

all_ratings = pd.concat(condition_ratings)
all_demos = pd.concat(demos)


# Create factor loadings
num_factors = 3
factor_names = ["factor" + str(i) for i in range(num_factors)]
exp_transformer = FactorAnalyzer(num_factors)
analysis = exp_transformer.fit(condition_ratings[0][question_names].to_numpy())
condition_ratings[0][factor_names] = exp_transformer.transform(condition_ratings[0][question_names].to_numpy())
condition_ratings[1][factor_names] = exp_transformer.transform(condition_ratings[1][question_names].to_numpy())
condition_ratings[2][factor_names] = exp_transformer.transform(condition_ratings[2][question_names].to_numpy())

"""model_spec = ModelSpecificationParser.parse_model_specification_from_dict(condition_ratings[question_names].to_numpy(),
                                                                          factor_structure)
cfa = ConfirmatoryFactorAnalyzer(model_spec, 42)
cfa_res = cfa.fit(condition_ratings[question_names])"""


# factor_score_weights = pd.DataFrame(analysis.loadings_.T, columns=question_names)
# We'll repeat this in a kind of full process cross validation
acc_by_factor = [[],[],[]]
mdn_metrics = [[],[],[]]
all_metrics = [[],[],[]]
for random_state in range(100):
    x, x_test, y, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i, data in enumerate(condition_ratings):
        x_n = data[["features", "uuid"]]
        y_n = data[question_names + factor_names].copy()

        if i == 0:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
            train_i, test_i = next(gss.split(x_n, y_n, groups=x_n["uuid"]))

            x = pd.concat([x, x_n.iloc[train_i]])
            x_test = pd.concat([x_test, x_n.iloc[test_i]])
            y = pd.concat([y, y_n.iloc[train_i]])
            y_test = pd.concat([y_test, y_n.iloc[test_i]])

        else:
            x = pd.concat([x, x_n])
            y = pd.concat([y, y_n])
        # Bin the data
        y_bin = y.copy()
        y_test_bin = y_test.copy()
        y_bin[question_names] = bin_likert(y[question_names].copy())
        y_test_bin[question_names] = bin_likert(y_test[question_names].copy())

        y_bin[factor_names] = bin_factor_score(y[factor_names].copy())
        y_test_bin[factor_names] = bin_factor_score(y_test[factor_names].copy())

        # Evaluate classification accuracy for each factor
        fac_acc = []
        for k, factor_name in enumerate(factor_names):
            clf = fit_svm(np.vstack(x["features"]), y_bin[factor_name])
            acc_by_factor[k].append(clf.score(np.vstack(x_test["features"]), y_test_bin[factor_name].to_numpy()))

        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
        train_i, val_i = next(gss.split(x, y, groups=x["uuid"]))
        x_train, y_train, x_val, y_val = x.iloc[train_i], y.iloc[train_i], x.iloc[val_i], y.iloc[val_i]

        name = f"mdn_{random_state}_{i}"
        print(name)
        mdn_res = train_mdn(x,y, x_val, y_val, x_test, y_test, name=name)

        mdn_metrics[i].append(mdn_res["test_loss"])
        all_metrics[i].append([mdn_res[f"test_loss_{i}"] for i in range(3)])
        print(mdn_metrics)
        print(all_metrics)

for i, acc in enumerate(acc_by_factor):
    acc_by_factor[i] = np.array(acc).reshape(-1,3)


from matplotlib import pyplot as plt, ticker

plt.errorbar([0,1,2],acc_by_factor[0].mean(0),yerr=acc_by_factor[0].std(0)/2, label="competent")
plt.errorbar([0,1,2],acc_by_factor[1].mean(0),yerr=acc_by_factor[1].std(0)/2, label="broken")
plt.errorbar([0,1,2],acc_by_factor[2].mean(0),yerr=acc_by_factor[2].std(0)/2, label="curious")
plt.gca().set_xlabel("Iteration")
plt.gca().set_ylabel("Accuracy")
plt.legend()
plt.gca().set_ylim((.2, .6))
plt.gca().set_xticks(np.arange(0,3))
plt.show()