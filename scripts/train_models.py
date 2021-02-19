import glob
import json
import os
import random
import shutil

import pandas as pd
import torch
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer

import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import TensorDataset

from models.mdn import mog_mode, mog_mean, mog_kl, ens_uncertainty_kl, ens_uncertainty_mode, ens_uncertainty_js, \
    ens_uncertainty_w
from models.nn import train_mdn
from models.plotting import make_mog
from models.simple import fit_svm, bin_factor_score, bin_likert
from models.util import process_turk_files, question_names
from search.util import load_map


def make_mog_test_plots(model, x_test, y_test, model_name=""):
    y_test = y_test[["factor0", "factor1", "factor2"]].to_numpy()
    test_data = TensorDataset(torch.from_numpy(np.vstack(x_test["features"])).float(), torch.from_numpy(y_test).float())
    pi, sigma, mu = model.forward(test_data.tensors[0])

    for i in range(len(x_test)):
        truth_i = x_test["uuid"] == x_test["uuid"].iloc[i]
        truth_y = y_test[truth_i]
        fig = make_mog(f"{model_name} traj={x_test['uuid'].iloc[i]}", pi[i], sigma[i], mu[i], true_points=truth_y)
        fig.show()
    return


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


def cross_validate_svm():
    acc_by_factor = [[], [], []]
    mdn_metrics = [[], [], []]
    all_metrics = [[], [], []]
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
            for k, factor_name in enumerate(factor_names):
                clf = fit_svm(np.vstack(x["features"]), y_bin[factor_name])
                acc_by_factor[k].append(clf.score(np.vstack(x_test["features"]), y_test_bin[factor_name].to_numpy()))

    for i, acc in enumerate(acc_by_factor):
        acc_by_factor[i] = np.array(acc).reshape(-1, 3)


def cross_validate_mdn():
    all_metrics = [[], [], []]
    for random_state in range(16):
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

            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
            train_i, val_i = next(gss.split(x, y, groups=x["uuid"]))
            x_train, y_train, x_val, y_val = x.iloc[train_i], y.iloc[train_i], x.iloc[val_i], y.iloc[val_i]

            name = f"mdn_{random_state}_{i}"
            print(name)
            mdn_res, best_model, _ = train_mdn(x, y, x_val, y_val, x_test, y_test, name=name)

            all_metrics[i].append(mdn_res[f"test_loss"])


            #make_mog_test_plots(best_model, x_test, y_test)
    print(np.array(all_metrics).mean(1))
    print("done")


def ensemble_mdn(n):
    all_metrics = []
    models = []
    x_all = pd.concat([data[["features", "uuid"]] for data in condition_ratings])
    y_all = pd.concat([data[question_names + factor_names].copy() for data in condition_ratings])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_i, test_i = next(gss.split(x_all, y_all, groups=x_all["uuid"]))

    x = x_all.iloc[train_i]
    x_test = x_all.iloc[test_i]
    y = y_all.iloc[train_i]
    y_test = y_all.iloc[test_i]

    y_test_np = y_test[["factor0", "factor1", "factor2"]].to_numpy()
    test_data = TensorDataset(torch.from_numpy(np.vstack(x_test["features"])).float(),
                              torch.from_numpy(y_test_np).float())
    for random_state in range(n):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
        train_i, val_i = next(gss.split(x, y, groups=x["uuid"]))
        x_train, y_train, x_val, y_val = x.iloc[train_i], y.iloc[train_i], x.iloc[val_i], y.iloc[val_i]

        name = f"mdn_{random_state}"
        print(name)
        mdn_res, best_model, best_path = train_mdn(x, y, x_val, y_val, x_test, y_test, name=name)
        shutil.copyfile(best_path, name+".ckpt")
        all_metrics.append(mdn_res[f"test_loss"])

        make_mog_test_plots(best_model, x_test, y_test)
        models.append(best_model)
    print("Metrics")
    print(np.array(all_metrics).mean(0))
    train_data = TensorDataset(torch.from_numpy(np.vstack(x["features"])).float(),
                               torch.from_numpy(np.vstack(y[["factor0", "factor1"]].to_numpy())).float())
    unc_js_train = ens_uncertainty_js(models, train_data.tensors[0])
    unc_js = ens_uncertainty_js(models, test_data.tensors[0])
    unc_mode_train = ens_uncertainty_mode(models, train_data.tensors[0])
    unc_mode = ens_uncertainty_mode(models, test_data.tensors[0])
    unc_train = ens_uncertainty_kl(models, train_data.tensors[0])
    unc = ens_uncertainty_kl(models, test_data.tensors[0])
    unc_w_train = ens_uncertainty_w(models, train_data.tensors[0])
    unc_w = ens_uncertainty_w(models, test_data.tensors[0])
    print("Train")
    print("kl",unc_train.mean(0))
    print("mode", unc_mode_train[1].mean(0))
    print("js", unc_js_train.mean(0))
    print("w", unc_w_train.mean(0))
    print("Test")
    print("kl",unc.mean(0))
    print("mode", unc_mode[1].mean(0))
    print("js", unc_js.mean(0))
    print("w", unc_w.mean(0))
    print("done")


def diff_featurize(plan, goal):
    points = np.array(plan, dtype=np.int)
    diffs_1 = np.diff(points, axis=0)
    diffs_2 = np.diff(points, 2, axis=0)
    diffs_2 = np.pad(diffs_2, ((1,0), (0,0)), mode="constant")
    in_goal = np.array([1 if p in goal else 0 for p in plan][1:]).reshape(-1, 1)
    return np.hstack([diffs_1,diffs_2, in_goal])

def fit_traj_lstm():
    all_metrics = []
    models = []
    modes = []
    means = []
    x_all = pd.concat([data[["trajectories", "uuid"]] for data in condition_ratings])

    trajectories = x_all["trajectories"].to_numpy()

    grid, bedroom = load_map("../interface/assets/map32.tmx")
    x_all["featurized"] = x_all["trajectories"].apply(lambda x: diff_featurize(x, bedroom))
    y_all = pd.concat([data[question_names + factor_names].copy() for data in condition_ratings])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_i, test_i = next(gss.split(x_all, y_all, groups=x_all["uuid"]))

    x = x_all.iloc[train_i]
    x_test = x_all.iloc[test_i]
    y = y_all.iloc[train_i]
    y_test = y_all.iloc[test_i]

    for random_state in range(50):

        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
        train_i, val_i = next(gss.split(x, y, groups=x["uuid"]))
        x_train, y_train, x_val, y_val = x.iloc[train_i], y.iloc[train_i], x.iloc[val_i], y.iloc[val_i]

        name = f"mdn_{random_state}"
        print(name)
        mdn_res, best_model = train_mdn(x, y, x_val, y_val, x_test, y_test, name=name)

        all_metrics.append([mdn_res[f"test_loss_{i}"] for i in range(3)])
        print(sum(all_metrics[-1]))

        y_test_np = y_test[["factor0", "factor1", "factor2"]].to_numpy()
        test_data = TensorDataset(torch.from_numpy(np.vstack(x_test["features"])).float(),
                                  torch.from_numpy(y_test_np).float())
        pi, sigma, mu = best_model.forward(test_data.tensors[0])

        mode = mog_mode(pi[0], sigma[0], mu[0])
        mean = mog_mean(pi[0], mu[0])
        modes.append(mode)
        means.append(mean.detach().numpy())
        make_mog_test_plots(best_model, x_test, y_test)
        # TODO: Check modes over multiple iterations. Calculate statistics on them. See if regular mean changes much
    mean_mode = np.array(modes).mean(0)
    std_mode = np.array(modes).std(0)
    print(mean_mode, std_mode)
    mean_mode = np.array(means).mean(0)
    std_mode = np.array(means).std(0)
    print(mean_mode, std_mode)
    print("done")

#fit_traj_lstm()
#cross_validate_mdn()
ensemble_mdn(16)
# factor_score_weights = pd.DataFrame(analysis.loadings_.T, columns=question_names)
# We'll repeat this in a kind of full process cross validation


