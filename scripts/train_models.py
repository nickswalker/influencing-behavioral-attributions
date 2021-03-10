import glob
import json
import os
import random
import shutil
import sys

import pandas as pd
import torch
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from joblib import dump
from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import IterativeImputer

import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import TensorDataset

from models.mdn import mog_mode, ens_uncertainty_kl, ens_uncertainty_mode, ens_uncertainty_js, \
    ens_uncertainty_w, marginal_mog_log_prob, ens_uncertainty_mean
from models.nn import train_mdn, MDNEnsemble
from models.plotting import make_mog
from models.simple import fit_svm, bin_factor_score, bin_likert
from models.util import process_turk_files
from models.util import question_names as all_question_names
from search.util import load_map


def make_mog_test_plots(model, x_test, y_test, model_name=""):
    y_test = y_test[["factor0", "factor1", "factor2"]].to_numpy()
    test_data = TensorDataset(torch.from_numpy(np.vstack(x_test["features"])).float(), torch.from_numpy(y_test).float())
    pi, sigma, mu = model.forward(test_data.tensors[0])

    n = 200
    x = np.linspace(-3, 3, n)
    x_batch = x.reshape([-1, 1])
    probs = torch.exp(marginal_mog_log_prob(pi, sigma, mu, torch.Tensor(x_batch))).detach().numpy()

    for i in range(len(x_test)):
        truth_i = x_test["uuid"] == x_test["uuid"].iloc[i]
        truth_y = y_test[truth_i]
        fig = make_mog(f"{model_name} traj={x_test['uuid'].iloc[i]}", probs[i], true_points=truth_y)
        fig.show()
    return


random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)

factor_structure = {"competent":
                        ["competent", "efficient", "focused", "intelligent", "reliable", "responsible"],
                    "broken": ["broken", "clumsy", "confused", "lost"],
                    "curious": ["curious", "investigative"]
                    }

condition_ratings = []
demos = []
other_data = []
max_id = 0
for base in ["pilot1", "active1", "active2", "mdn_active1", "mdn_active2"]:
    cr, d, o = process_turk_files(base + ".csv", traj_file=base + "_trajs.json")
    q_names = [q_name for q_name in all_question_names if q_name in cr.columns]
    # if "inquisitive" in cr.columns:
    #    cr.drop(columns=["inquisitive"], inplace=True)
    # Fill in missing values
    cr[cr[q_names] == 6] = np.nan
    imp = IterativeImputer(missing_values=np.nan, max_iter=200, random_state=0, min_value=1, max_value=5)
    to_impute = cr[q_names].to_numpy()
    cr[q_names] = np.rint(imp.fit_transform(to_impute)).astype(int)
    cr["uuid"] = cr["id"] + max_id
    max_id = cr["uuid"].max() + 1
    assert not cr[cr[q_names] == np.nan].any().any()
    assert not cr[cr[q_names] == 6].any().any()
    condition_ratings.append(cr)
    demos.append(d)
    other_data.append(o)

all_ratings = pd.concat(condition_ratings)
all_demos = pd.concat(demos)


# Create factor loadings
# Original, first SVM pilot factors
num_factors = 3
first_step_data = condition_ratings[0]
first_q_names = [q_name for q_name in all_question_names if q_name in first_step_data.columns]
factor_names = ["factor" + str(i) for i in range(num_factors)]
exp_transformer = FactorAnalyzer(n_factors=num_factors, rotation="promax",)
analysis = exp_transformer.fit(first_step_data[first_q_names].to_numpy())

# Analysis across three SVM pilot factor
exp_transformer2 = FactorAnalyzer(n_factors=4, rotation="promax")
analysis2 = exp_transformer2.fit(pd.concat(condition_ratings[:3])[first_q_names].to_numpy())

exp_transformer3 = FactorAnalyzer(n_factors=num_factors, rotation="promax")
analysis3 = exp_transformer3.fit(pd.concat(condition_ratings[:3])[first_q_names].to_numpy())

exp_transformer34 = FactorAnalyzer(n_factors=4, rotation="promax")
analysis34 = exp_transformer34.fit(pd.concat(condition_ratings[:3])[first_q_names].to_numpy())

exp_transformer35 = FactorAnalyzer(n_factors=5, rotation="promax")
analysis35 = exp_transformer35.fit(pd.concat(condition_ratings[:3])[first_q_names].to_numpy())

small_question_names = first_q_names.copy()
small_question_names.remove("lazy")


exp_transformer3s = FactorAnalyzer(n_factors=num_factors, rotation="promax")
analysis3s = exp_transformer3s.fit(pd.concat(condition_ratings[:3])[small_question_names].to_numpy())

small_question_names.remove("energetic")


exp_transformer3ss = FactorAnalyzer(n_factors=num_factors, rotation="promax")
analysis3ss = exp_transformer3ss.fit(pd.concat(condition_ratings[:3])[small_question_names].to_numpy())

exp_transformer3ss4 = FactorAnalyzer(n_factors=4, rotation="promax")
analysis3ss4 = exp_transformer3ss4.fit(pd.concat(condition_ratings[:3])[small_question_names].to_numpy())

exp_transformer3ss5 = FactorAnalyzer(n_factors=5, rotation="promax")
analysis3ss5 = exp_transformer3ss5.fit(pd.concat(condition_ratings[:3])[small_question_names].to_numpy())


# With new data (inquisitive added)
exp_transformer4 = FactorAnalyzer(n_factors=4, rotation="promax")
analysis4 = exp_transformer4.fit(pd.concat(condition_ratings[3:])[all_question_names].to_numpy())

# New data, lazy dropped
new_small_question_names = all_question_names.copy()
new_small_question_names.remove("lazy")
exp_transformer4s = FactorAnalyzer(n_factors=3, rotation="promax")
analysis4s = exp_transformer4s.fit(pd.concat(condition_ratings[3:])[new_small_question_names].to_numpy())

new_small_question_names.remove("energetic")

exp_transformer4ss = FactorAnalyzer(n_factors=3, rotation="promax")
analysis4ss = exp_transformer4ss.fit(pd.concat(condition_ratings[3:])[new_small_question_names].to_numpy())

chosen_transformer = exp_transformer3ss
question_names = small_question_names
dump(chosen_transformer, "factor_model.pickle")

for i in range(len(condition_ratings)):
    condition_ratings[i][factor_names] = chosen_transformer.transform(condition_ratings[i][question_names].to_numpy())


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


def cv_mdn_stepwise(n, ens_size, datasteps, hparams, chosen_test=None, dry_run=True, select_with_test=False):
    print(n, ens_size, hparams)

    # Datastep x N x ens size
    ind_test_loss = np.empty([len(datasteps), n, ens_size])
    ind_val_loss = np.empty([len(datasteps), n, ens_size])
    # Datastep x N
    ens_test_loss = np.empty([len(datasteps), n])
    ens_val_loss = np.empty([len(datasteps), n])

    for random_state in range(n):
        x0, y0 = datasteps[0][["features", "uuid"]], datasteps[0][factor_names]
        if chosen_test:
            # Pull the test indices, which are presumed to be from the first datastep
            all_indices = list(range(len(x0)))
            nontest_indices = [i for i in all_indices if i not in chosen_test]
            x_tv, y_tv, x_test, y_test = x0.iloc[nontest_indices], y0.iloc[nontest_indices], x0.iloc[chosen_test], y0.iloc[chosen_test]
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
            train_i, test_i = next(gss.split(x0, y0, groups=x0["uuid"]))
            x_tv, y_tv, x_test, y_test = x0.iloc[train_i], y0.iloc[train_i], x0.iloc[test_i], y0.iloc[test_i]

        x_test_ten, y_test_ten = torch.from_numpy(np.vstack(x_test["features"])).float(), torch.from_numpy(y_test[factor_names].to_numpy()).float()

        for i, data in enumerate(datasteps):
            x_n = data[["features", "uuid"]]
            y_n = data[question_names + factor_names].copy()

            # Later batches are just more train data. Test was fixed from first set
            if i != 0:
                x_tv = pd.concat([x_tv, x_n])
                y_tv = pd.concat([y_tv, y_n])

            ens_models = []
            ens_checkpoints = []
            for j in range(ens_size):
                # In the case that the test set is fixed, train-val split is the only external randomness,
                # so to make sure we can average over all results and have independent noise, we'll
                # make sure that the split varies for each.
                gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=j + random_state * j)
                train_i, val_i = next(gss.split(x_tv, y_tv, groups=x_tv["uuid"]))
                x_train, y_train, x_val, y_val = x_tv.iloc[train_i], y_tv.iloc[train_i], x_tv.iloc[val_i], y_tv.iloc[
                    val_i]
                name = f"mdn_bag_{random_state}_{i}_{j}"
                print(name)
                mdn_res, best_model, best_checkpoint_path = train_mdn(x_train, y_train, x_val, y_val, x_test, y_test, hparams=hparams,
                                                   name=name)

                ind_test_loss[i][random_state][j] = mdn_res[f"test_loss"]
                ind_val_loss[i][random_state][j] = mdn_res[f"val_loss"]
                ens_models.append(best_model)
                ens_checkpoints.append(best_checkpoint_path)
            ens = MDNEnsemble(ens_models)
            ens.freeze()

            test_loss = ens.mean_nll(x_test_ten.to(device=best_model.device), y_test_ten.to(device=best_model.device))
            ens_test_loss[i][random_state] = test_loss
            ens_val_loss[i][random_state] = ind_val_loss[i,random_state,:].mean()
            print(f"Ens{random_state}: {test_loss}")
            if not select_with_test:
                criteria = ens_val_loss[i,:random_state + 1]
            else:
                criteria = ens_test_loss[i,:random_state + 1]
            if not dry_run and i == (len(datasteps) - 1) and criteria.argmin() == random_state:
                print(f"New best loss {criteria[random_state]}")
                for k, path in enumerate(ens_checkpoints):
                    shutil.copyfile(path, f"mdn_{k}" + ".ckpt")

        # First model of last round for each datastep
        first_m_res = ind_test_loss[:,random_state, 0]
        ind_all = ind_test_loss[:,random_state,:].reshape(len(datasteps), -1)
        print(f"N=1", first_m_res)
        print(f"N={ind_all.shape[-1]}", ind_all.mean(1), ind_all.std(1))
        # Get last ensemble and report result
        print(f"Ens={ens_size}, N={1}", ens_test_loss[:,random_state])

        #make_mog_test_plots(best_model, x_test, y_test)
    ind_test_loss = ind_test_loss.reshape(len(datasteps), -1)
    print(f"N={ind_test_loss.shape[-1]}", ind_test_loss.mean(-1), ind_test_loss.std(-1))
    print(f"N={ens_test_loss.shape[-1]}", ens_test_loss.mean(-1), ens_test_loss.std(-1))
    print("done")
    return (ind_test_loss.mean(-1), ind_test_loss.std(-1)), (ens_test_loss.mean(-1), ens_test_loss.std(-1))


def ensemble_mdn_unc(ens_size, hparams, dry_run=True):
    print(hparams)
    all_metrics = []
    models = []
    x_all = pd.concat([data[["features", "uuid"]] for data in condition_ratings])
    y_all = pd.concat([data[question_names + factor_names].copy() for data in condition_ratings])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    tv_i, test_i = next(gss.split(x_all, y_all, groups=x_all["uuid"]))
    if not dry_run:
        with open("test_i.txt", 'w') as f:
            f.write(str(test_i.tolist()))
            f.write(str(list(set(x_all.iloc[test_i]["uuid"].to_list()))))
    x, y = x_all.iloc[tv_i],  y_all.iloc[tv_i]
    x_test,   y_test = x_all.iloc[test_i], y_all.iloc[test_i]

    y_test_np = y_test[["factor0", "factor1", "factor2"]].to_numpy()
    test_data = TensorDataset(torch.from_numpy(np.vstack(x_test["features"])).float(),
                              torch.from_numpy(y_test_np).float())
    for random_state in range(ens_size):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
        train_i, val_i = next(gss.split(x, y, groups=x["uuid"]))
        x_train, y_train, x_val, y_val = x.iloc[train_i], y.iloc[train_i], x.iloc[val_i], y.iloc[val_i]

        name = f"mdn_{random_state}"
        print(name)
        mdn_res, best_model, best_path = train_mdn(x, y, x_val, y_val, x_test, y_test, name=name, hparams=hparams)
        if not dry_run:
            shutil.copyfile(best_path, name+".ckpt")
        all_metrics.append(mdn_res[f"test_loss"])

        #make_mog_test_plots(best_model, x_test, y_test)
        models.append(best_model)

    ens = MDNEnsemble(models)
    ens.freeze()

    print("Metrics")
    all = np.array(all_metrics)
    print(f"N={ens_size}", all.mean(0), all.std(0))
    print(f"Ens={ens_size}", ens.mean_nll(test_data.tensors[0], test_data.tensors[1]))
    train_data = TensorDataset(torch.from_numpy(np.vstack(x["features"])).float(),
                               torch.from_numpy(np.vstack(y[["factor0", "factor1"]].to_numpy())).float())
    #unc_js_train = ens_uncertainty_js(ens, train_data.tensors[0])
    #unc_js = ens_uncertainty_js(ens, test_data.tensors[0])
    unc_mode_train = ens_uncertainty_mode(ens, train_data.tensors[0])
    unc_mode = ens_uncertainty_mode(ens, test_data.tensors[0])
    unc_mean_train = ens_uncertainty_mean(ens, train_data.tensors[0])
    unc_mean = ens_uncertainty_mean(ens, test_data.tensors[0])
    unc_train = ens_uncertainty_kl(ens, train_data.tensors[0])
    unc = ens_uncertainty_kl(ens, test_data.tensors[0])
    #unc_w_train = ens_uncertainty_w(ens, train_data.tensors[0])
    #unc_w = ens_uncertainty_w(ens, test_data.tensors[0])
    print("Train")
    print("kl",unc_train.mean(0))
    print("mode", unc_mode_train.mean(0))
    print("mean", unc_mean_train.mean(0))
    #print("js", unc_js_train.mean(0))
    #print("w", unc_w_train.mean(0))
    print("Test")
    print("kl",unc.mean(0))
    print("mode", unc_mode.mean(0))
    print("mean", unc_mean.mean(0))
    #print("js", unc_js.mean(0))
    #print("w", unc_w.mean(0))
    print("done")


def train_val_test_split(x, y, groups, test_size=0.3, val_size=0.3, random_state=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_i, test_i = next(gss.split(x, y, groups=groups))

    val_gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_i, val_i = next(val_gss.split(x.iloc[train_i], y[train_i], groups=groups.iloc[train_i]))
    return train_i, val_i, test_i


def cv_bagged_ensemble_mdn(n, ens_size, hparams):
    print(n, ens_size, hparams)
    all_metrics = []
    ens_metrics = []
    x_all = pd.concat([data[["features", "uuid"]] for data in condition_ratings])
    y_all = pd.concat([data[question_names + factor_names].copy() for data in condition_ratings])
    for random_state in range(n):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
        tv_i, test_i = next(gss.split(x_all, y_all, groups=x_all["uuid"]))
        x_tv, y_tv, x_test, y_test = x_all.iloc[tv_i], y_all.iloc[tv_i], x_all.iloc[test_i], y_all.iloc[test_i]

        x_test_ten, y_test_ten = torch.from_numpy(np.vstack(x_test["features"])).float(), torch.from_numpy(y_test[["factor0", "factor1", "factor2"]].to_numpy()).float()

        all_metrics.append([])
        ens_models = []
        for i in range(ens_size):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=i)
            train_i, val_i = next(gss.split(x_tv, y_tv, groups=x_tv["uuid"]))
            x_train, y_train, x_val, y_val = x_tv.iloc[train_i], y_tv.iloc[train_i], x_tv.iloc[val_i], y_tv.iloc[val_i]
            name = f"mdn_bag_{random_state}_{i}"
            print(name)
            mdn_res, best_model, _ = train_mdn(x_train, y_train, x_val, y_val, x_test, y_test, hparams=hparams, name=name)

            all_metrics[-1].append(mdn_res[f"test_loss"])
            ens_models.append(best_model)
        ens = MDNEnsemble(ens_models)
        ens.freeze()

        test_loss = ens.mean_nll(x_test_ten, y_test_ten)
        ens_metrics.append(test_loss)
        print(f"Ens{random_state}: {test_loss}")

    first_m_res = np.array(all_metrics)[:, 0]
    print(f"N={n}", first_m_res.mean(0), first_m_res.std(0))
    print(f"Ens={ens_size}, N={n}", np.array(ens_metrics).mean(0), np.array(ens_metrics).std(0))


def percentage_data_experiment(n, ens_size, data, hparams):
    print(n, ens_size, hparams)
    percentages = np.linspace(0.1, 1.0, 10).tolist()
    ind_res = np.empty([len(percentages),n])
    ens_res = np.empty([len(percentages), n])
    for random_state in range(n):
        # Let's take the test data off the top
        x_all = data[["features", "uuid"]]
        y_all = data[question_names + factor_names].copy()
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
        tv_i, test_i = next(gss.split(x_all, y_all, groups=x_all["uuid"]))
        x_tv, y_tv, x_test, y_test = x_all.iloc[tv_i], y_all.iloc[tv_i], x_all.iloc[test_i], y_all.iloc[test_i]
        merged_test = pd.concat([x_test, y_test], axis=1)
        for i, train_percent in enumerate(percentages):
            if (1.0 - train_percent) != 0.0:
                gss = GroupShuffleSplit(n_splits=1, test_size=1.0 - train_percent, random_state=random_state)
                reduced_tv_i, _ = next(gss.split(x_tv, y_tv, groups=x_tv["uuid"]))
            else:
                reduced_tv_i = list(range(len(x_tv)))
            red_x_tv, red_y_tv = x_tv.iloc[reduced_tv_i], y_tv.iloc[reduced_tv_i]
            reduced_data = pd.concat([red_x_tv,red_y_tv], axis=1)
            test_start_i = len(reduced_data)
            reduced_data_w_test = pd.concat([reduced_data, merged_test])
            new_test_i = list(range(test_start_i, len(reduced_data_w_test)))

            res = cv_mdn_stepwise(1, ens_size, [reduced_data_w_test], hparams, chosen_test=new_test_i)
            ind_res[i,random_state] = res[0][0].item()
            ens_res[i,random_state] = res[1][0].item()
        print(f"finished {train_percent}")

    print("ind",ind_res.mean(-1), ind_res.std(-1))
    print("ens",ens_res.mean(-1), ens_res.std(-1))
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

from ast import literal_eval
with open("ens1/test_i.txt") as f:
    test_indices = literal_eval(f.readline())

#percentage_data_experiment(8, 8, pd.concat(condition_ratings), {"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15})
#percentage_data_experiment(8, 8, pd.concat(condition_ratings), {"hidden_size": 5, "gaussians": 1, "noise_regularization": 0.15})
percentage_data_experiment(8, 1, pd.concat(condition_ratings), {"hidden_size": 5, "gaussians": 1, "noise_regularization": 0.15})

sys.exit(0)

#cv_mdn_stepwise(8, 8, [pd.concat(condition_ratings)], {"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15}, dry_run=False, select_with_test=True)
#sys.exit(0)

print("STEPWISE START")
datasteps = [pd.concat(condition_ratings[:3])] + condition_ratings[3:]
cv_mdn_stepwise(8, 8, datasteps, {"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15},chosen_test=test_indices, dry_run=False)
cv_mdn_stepwise(8, 8, datasteps, {"hidden_size": 5, "gaussians": 1, "noise_regularization": 0.15},chosen_test=test_indices)
print("STEPWISE END")
#ensemble_mdn_unc(16, {"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15}, dry_run=False)
#sys.exit(0)





cv_bagged_ensemble_mdn(16,4,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15})
cv_bagged_ensemble_mdn(16,8,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15})
cv_bagged_ensemble_mdn(16,16,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15})


#cv_bagged_ensemble_mdn(8,8,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15})
cv_bagged_ensemble_mdn(8,8,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.05, "early_stopping": 500})
cv_bagged_ensemble_mdn(8,8,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.10, "early_stopping": 500})
cv_bagged_ensemble_mdn(8,1,{"hidden_size": 5, "gaussians": 1, "noise_regularization": 0.15, "early_stopping": 500})
cv_bagged_ensemble_mdn(8,8,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15, "early_stopping": 500})


#cv_bagged_ensemble_mdn(64,1,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15})

cv_bagged_ensemble_mdn(8,8,{"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.10})

#fit_traj_lstm()
#cross_validate_mdn(16, {"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15})
#cross_validate_mdn(16, {"hidden_size": 5, "gaussians": 1, "noise_regularization": 0.15})

cv_mdn_stepwise(16, {"hidden_size": 5, "gaussians": 3, "noise_regularization": 0.15})
cv_mdn_stepwise(16, {"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.10})
#cross_validate_mdn(16, {"hidden_size": 5, "gaussians": 1, "noise_regularization": 0.10})
cv_mdn_stepwise(16, {"hidden_size": 5, "gaussians": 3, "noise_regularization": 0.10})

# factor_score_weights = pd.DataFrame(analysis.loadings_.T, columns=question_names)
# We'll repeat this in a kind of full process cross validation


