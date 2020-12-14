import glob
import math
import re

import pandas as pd
import sklearn
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load

from matplotlib.backends.backend_pdf import PdfPages

import ast

question_code_to_name = {"b1": "broken", "b2": "clumsy", "b3": "competent", "b4": "confused", "b5": "curious",
                         "b6": "efficient", "b7": "energetic", "b8": "focused", "b9": "intelligent", "b10": "investigative", "b11": "lazy", "b12": "lost", "b13": "reliable", "b14": "responsible"}
question_names = sorted(list(question_code_to_name.values()))
feature_names = ["goal_cov",
                 "overlap",
                 "length",
                 "straight template match",
                 "hook template match",
                 "carpet time",
                 "collision time",
                 "redundant coverage",
                 "total coverage",
                 "idle time",
                 "start-stop template match"
                 ]
feats = [[1.0, 0.2911392405063291, 0.5602836879432624, 0.6363636363636364, 0.07792207792207792, 0.2978723404255319, 0.0, 0.23404255319148937, 0.09507640067911714, 0.0, 0.02564102564102564], [1.0, 0.41509433962264153, 0.75177304964539, 0.46153846153846156, 0.25961538461538464, 0.3829787234042553, 0.0, 0.10638297872340426, 0.10526315789473684, 0.2571428571428571, 0.09523809523809523], [0.25531914893617025, 0.5316455696202531, 0.5602836879432624, 0.23376623376623376, 0.6623376623376623, 0.5531914893617021, 0.0, 0.1702127659574468, 0.06281833616298811, 0.0, 0.02564102564102564], [1.0, 0.5714285714285714, 1.0, 0.35526315789473684, 0.3355263157894737, 0.723404255319149, 0.0, 0.723404255319149, 0.11205432937181664, 0.0, 0.013071895424836602], [0.3829787234042553, 0.07692307692307687, 0.2765957446808511, 0.5405405405405406, 0.16216216216216217, 0.14893617021276595, 0.0, 0.0425531914893617, 0.06112054329371817, 0.0, 0.05263157894736842], [0.0, 0.6153846153846154, 0.2765957446808511, 0.6216216216216216, 0.0, 0.0, 0.0, 0.0, 0.025466893039049237, 0.5526315789473685, 0.42105263157894735], [0.0, 0.4772727272727273, 0.3120567375886525, 0.7142857142857143, 0.07142857142857142, 0.5106382978723404, 0.0, 0.0, 0.03904923599320883, 0.3488372093023256, 0.18604651162790697], [0.9574468085106383, 0.3370786516853933, 0.6312056737588653, 0.5977011494252874, 0.06896551724137931, 0.3617021276595745, 0.0425531914893617, 0.425531914893617, 0.100169779286927, 0.0, 0.022727272727272728], [0.8936170212765957, 0.29761904761904767, 0.5957446808510638, 0.6097560975609756, 0.07317073170731707, 0.10638297872340426, 0.0, 0.2553191489361702, 0.100169779286927, 0.012048192771084338, 0.04819277108433735], [0.9787234042553191, 0.3142857142857143, 0.7446808510638298, 0.4368932038834951, 0.2621359223300971, 0.3829787234042553, 0.0, 0.10638297872340426, 0.12224108658743633, 0.23076923076923078, 0.09615384615384616], [0.8723404255319149, 0.4579439252336449, 0.7588652482269503, 0.47619047619047616, 0.17142857142857143, 0.40425531914893614, 0.0, 0.06382978723404255, 0.09847198641765705, 0.3584905660377358, 0.16981132075471697], [0.23404255319148937, 0.5189873417721519, 0.5602836879432624, 0.23376623376623376, 0.8571428571428571, 0.574468085106383, 0.0, 0.14893617021276595, 0.06451612903225806, 0.0, 0.02564102564102564], [0.25531914893617025, 0.5568181818181819, 0.624113475177305, 0.22093023255813954, 0.6627906976744186, 0.5106382978723404, 0.19148936170212766, 0.19148936170212766, 0.06621392190152801, 0.05747126436781609, 0.06896551724137931], [1.0, 0.5714285714285714, 1.0, 0.35526315789473684, 0.3355263157894737, 0.723404255319149, 0.0, 0.723404255319149, 0.11205432937181664, 0.0, 0.013071895424836602], [1.0, 0.6023391812865497, 1.0, 0.33727810650887574, 0.31952662721893493, 0.7446808510638298, 0.0, 1.0, 0.11544991511035653, 0.0058823529411764705, 0.023529411764705882], [0.574468085106383, 0.11764705882352944, 0.3617021276595745, 0.5102040816326531, 0.1836734693877551, 0.19148936170212766, 0.0, 0.10638297872340426, 0.07640067911714771, 0.0, 0.04], [0.3829787234042553, 0.16049382716049387, 0.574468085106383, 0.5189873417721519, 0.1518987341772152, 0.14893617021276595, 0.02127659574468085, 0.0425531914893617, 0.11544991511035653, 0.0, 0.025], [0.0, 0.6060606060606061, 0.23404255319148937, 0.5483870967741935, 0.1935483870967742, 0.0, 0.0, 0.0, 0.022071307300509338, 0.46875, 0.375], [0.0, 0.625, 0.28368794326241137, 0.6052631578947368, 0.0, 0.02127659574468085, 0.0, 0.0, 0.025466893039049237, 0.358974358974359, 0.41025641025641024], [0.0, 0.6744186046511628, 0.3049645390070922, 0.7073170731707317, 0.07317073170731707, 0.5106382978723404, 0.0, 0.0, 0.023769100169779286, 0.38095238095238093, 0.19047619047619047], [0.0, 0.28125, 0.22695035460992907, 0.6666666666666666, 0.1, 0.2553191489361702, 0.0, 0.0, 0.03904923599320883, 0.16129032258064516, 0.1935483870967742]]
feats = np.stack(feats)

factor_structure = {"competent":
                        ["competent", "efficient", "energetic", "focused", "intelligent", "reliable", "responsible"],
                    "broken": ["broken", "clumsy", "confused", "lazy", "lost"],
                   "curious": ["curious", "investigative"]
                    }

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
        plt.show()
        return fig


def make_fa_plots(grouped):
    # Run factor analysis by condition
    num_factors = 4
    i = 0
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
    for c_name, group in grouped:
        row = math.floor(i / 2)
        column = i % 2
        transformer = FactorAnalyzer(num_factors)
        analysis = transformer.fit(group[question_code_to_name.values()].to_numpy())
        plot_components(analysis.loadings_, axes[row][column], list(question_code_to_name.values()), c_name,
                        title_prefix="FA ")
        i += 1
        """
        # scikit's FA doesn't have rotations, so it's harder to interpret
        transformer = FactorAnalysis(num_factors)
        analysis = transformer.fit(group.to_numpy())
        plot_components(analysis.components_.transpose(), title_prefix="FA ")
        """
    plt.show()
    return fig


def make_fa_plots(data, analysis):
    # Run factor analysis by condition
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
    plot_components(analysis.loadings_, ax, list(question_code_to_name.values()), "All", title_prefix="FA ")

    """
    # scikit's FA doesn't have rotations, so it's harder to interpret
    transformer = FactorAnalysis(num_factors)
    analysis = transformer.fit(group.to_numpy())
    plot_components(analysis.components_.transpose(), title_prefix="FA ")
    """
    plt.show()
    return fig


def fit_lin_reg(condition_ratings):
    print("Linear regression targeting items -------------------")
    scaler = StandardScaler()
    scaler.fit(condition_ratings[question_names].to_numpy())

    scaled = scaler.transform(condition_ratings[question_names])

    y = scaled
    x = np.vstack(condition_ratings["features"])

    model = linear_model.LinearRegression()
    model.fit(x, y)
    # print(model.coef_)
    # print(model.intercept_)
    print("lin reg R2 total\t\t {:.2}".format(model.score(x, y)))

    for q_name in question_names:
        y = condition_ratings[q_name].to_numpy()

        model = linear_model.LinearRegression()
        model.fit(x, y)
        # print(model.coef_)
        # print(model.intercept_)
        print("lin reg R2 {} \t\t{:.2}".format(q_name, model.score(x, y)))

    print("Linear regression targeting factors -------------------")
    factor_names = ["factor"+str(i) for i in range(3)]
    y = condition_ratings[factor_names].to_numpy()
    x = np.vstack(condition_ratings["features"])

    model = linear_model.LinearRegression()
    model.fit(x, y)
    # print(model.coef_)
    # print(model.intercept_)
    print("linear model factors R2 total\t\t {:.2}".format(model.score(x, y)))

    for factor in factor_names:
        y = condition_ratings[factor].to_numpy()

        model = linear_model.LinearRegression()
        model.fit(x, y)
        # print(model.coef_)
        # print(model.intercept_)
        print("lin reg R2 {} \t\t{:.2}".format(factor, model.score(x, y)))

    # http://proceedings.mlr.press/v89/paananen19a/paananen19a.pdf
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   random_state=0).fit(x, y)
    print("Gausian process R2: {}".format(gpr.score(x, y)))

    # print(gpr.predict(x[:1, :], return_std=True))


def fit_classification(condition_ratings):
    print("Classification over features targeting items---------------")
    y = condition_ratings[question_names].copy()
    y[y < 3] = 0
    y[y == 3] = 1
    y[y > 3] = 2

    x = np.vstack(condition_ratings["features"])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
    for i, attr_name in enumerate(question_names):
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, y_train[attr_name].to_numpy())
        print("svm acc {} \t\t {:.2}".format(attr_name, clf.score(X_test, y_test[attr_name].to_numpy())))

    print("Classification over features targeting factors---------------")
    factor_names = ["factor" + str(i) for i in range(3)]
    y = condition_ratings[factor_names].copy()
    x = np.vstack(condition_ratings["features"])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
    factor_clfs = []
    conf_mats = []
    y_tests = []
    for factor in factor_names:
        train = y_train[factor].copy()
        low = train < -.5
        mid = (-.5 < train) & (train < .5)
        high = .5 < train
        train[low] = 0
        train[mid] = 1
        train[high] = 2
        train = train.astype(int)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, train.to_numpy())
        test = y_test[factor].copy()
        low = test < -.5
        mid = (-.5 < test) & (test < .5)
        high = .5 < test

        test[low] = 0
        test[mid] = 1
        test[high] = 2
        test = test.astype(int)
        test.reset_index(None, drop=True, inplace=True)
        print("svm acc {} \t\t {:.2}".format(factor, clf.score(X_test, test.to_numpy())))

        conf_mats.append(confusion_matrix(test, clf.predict(X_test), sample_weight=None, normalize=None))
        y_tests.append(test)
        factor_clfs.append(clf)
    conf_plots = make_conf_mat_plots(factor_clfs, X_test, y_tests)
    dump(factor_clfs, "factor_svms.pickle")
    return factor_clfs, conf_plots


def process_turk_files(paths, filter_rejected=True):
    print("Processing paths: {}".format(str(paths)))

    def drop_trailing_num(name):
        try:
            return next(re.finditer(r'[\D\.]*', name)).group(0)
        except StopIteration:
            return name

    def drop_con_number(name):
        try:
            con_string = next(re.finditer(r'con_\d_', name)).group(0)
            return name.replace(con_string, "")
        except StopIteration:
            return name

    def drop_id_trailing_number(name):
        try:
            return next(re.finditer(r'Input.id', name)).group(0)
        except StopIteration:
            return name

    def drop_aid_trailing_number(name):
        try:
            return next(re.finditer(r'Input.aid', name)).group(0)
        except StopIteration:
            return name

    # Ignore index so that we get a new ID per row, instead of numbers by file
    frame = pd.concat([pd.read_csv(path, na_filter=True) for path in paths], ignore_index=True)
    if filter_rejected:
        frame.drop(frame[frame["AssignmentStatus"] == "Rejected"].index, inplace=True)
    submitted = len(frame[frame["AssignmentStatus"] == "Submitted"])
    if submitted > 0:
        print("{} still pending".format(submitted))
    frame.drop(frame.columns[frame.columns.str.contains(r"noop")], axis=1, inplace=True)
    grouped = frame.groupby(frame.columns.str.extract(r"Answer.con_(\d+)_.*", expand=False), axis=1)
    # Decode the one hot representation of scale responses
    for condition_name, group in grouped:
        questions = group.groupby(group.columns.str.extract(r"Answer.con_\d+_(.*).yes\d+", expand=False),
                                  axis=1)
        for q_name, question in questions:
            summary_col_name = "Answer.con_{}_{}".format(condition_name, q_name)
            frame[summary_col_name] = (frame[question.columns] == 1).idxmax(axis=1).str.extract(r"(\d+)$").astype(int)

    frame.drop(frame.columns[frame.columns.str.contains("yes")], axis=1, inplace=True)

    condition_views = []
    # Count the occurences of Input.id. Sometimes we issue batches with different numbers
    num_conditions = frame.filter(regex='^Input.id', axis=1).shape[1]
    for n in range(num_conditions):
        answers_cols = list(frame.columns[frame.columns.str.contains("con_" + str(n))])
        answers_cols = list(filter(lambda x: "traj" not in x, answers_cols))
        columns = ["Input.id" + str(n), "WorkerId"] + answers_cols
        condition_views.append(frame[columns].rename(columns=drop_con_number).rename(columns=drop_id_trailing_number))
    condition_ratings = pd.concat(condition_views, ignore_index=True)
    # We loaded many different files. Some may not have as many command columns as others
    condition_ratings.dropna(inplace=True)

    demo_views = []
    # Count the occurences of Input.id. Sometimes we issue batches with different numbers
    num_demos = frame.filter(regex='^Input.aid', axis=1).shape[1]
    for n in range(num_demos):
        answers_cols = list(frame.columns[frame.columns.str.contains("con_" + str(n))])
        answers_cols = list(filter(lambda x: "traj" in x, answers_cols))
        columns = ["Input.aid" + str(n), "WorkerId"] + answers_cols
        demo_views.append(frame[columns].rename(columns=drop_con_number).rename(columns=drop_aid_trailing_number))
    demos = pd.concat(demo_views, ignore_index=True)
    # We loaded many different files. Some may not have as many command columns as others
    demos.dropna(inplace=True)

    # Drop all the extra input or answer fields except those you want to use for additional stats
    other_data = frame.drop(
        columns=[c for c in frame.columns if
                 ("Input" in c or "Answer" in c) and c != "Answer.comment" and c != "Input.hitid"])

    # Weird behavior on explain column if I do "Answer."
    condition_ratings.columns = condition_ratings.columns.str.lstrip(r'Answer')
    condition_ratings.columns = condition_ratings.columns.str.lstrip(r'.')
    condition_ratings.rename(columns=question_code_to_name, inplace=True)
    condition_ratings.columns = condition_ratings.columns.str.lstrip("Input.")

    demos.columns = demos.columns.str.lstrip("Answer.")
    demos.columns = demos.columns.str.lstrip("Input.")
    return condition_ratings, demos, other_data

condition_ratings, demos, other_data = process_turk_files(glob.glob("pilot1.csv"))

demo_trajs = list(demos["raj"].apply(lambda x: ast.literal_eval(x)))
#demo_trajs = list(demos["raj"])
demo_exps = list(demos["xplain_traj"])
demo_prompts = list(demos["aid"])
demo_wids = list(demos["WorkerId"])

"""print("DEMOS-------------")
print(demo_trajs)
print(demo_prompts)
print(demo_exps)
print(demo_wids)
print("DEMO END---------------------")"""

# Add feats and traj data
condition_ratings["features"] = condition_ratings["id"].apply(lambda x: feats[x]).to_numpy()
condition_ratings["trajectories"] = condition_ratings["id"].apply(lambda x: feats[x]).to_numpy()

# Fill in missing values
imp = IterativeImputer(missing_values=6, max_iter=10, random_state=0)
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

factor_score_weights = pd.DataFrame(analysis.loadings_.T,columns=question_names)
factor_score_weights[factor_score_weights > .4] = 1
factor_score_weights[factor_score_weights <= .4] = 0

factor_names = ["factor" + str(i) for i in range(num_factors)]
condition_ratings[factor_names] = transformed
for i in range(num_factors):
    num_components = factor_score_weights.iloc[i].sum()
    condition_ratings["afactor"+str(i)] = (condition_ratings[question_names] * factor_score_weights.iloc[i]).sum(axis=1) /num_components

fit_lin_reg(condition_ratings)
conf_mats, conf_plots = fit_classification(condition_ratings)

by_condition = condition_ratings.groupby("id")
cond_qual = []
for i in range(21):
    cond_qual.append([])
for name, group in by_condition:
    #print(name)
    for _, (describe, explain, wid) in group[["describe", "explain", "WorkerId"]].iterrows():
        #print(describe)
        #print(explain)
       # print("")
        cond_qual[int(name)].append(describe + "---" + explain + "--" + wid)

#print(cond_qual)
turker_performance = pd.DataFrame()
turker_performance["HITTime"] = other_data.groupby("WorkerId")["WorkTimeInSeconds"].mean()
# turker_performance["Comment"] = other_data.groupby("WorkerId")["Answer.comment"].apply(lambda x: ','.join(x))


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
    for _, (aid, explain) in group[["aid","xplain_traj"]].iterrows():
        print(aid)
        print(explain)
    print("")
       #print("")

turker_performance.to_csv("turker_stats.txt", index=False)

# Histograms
num_participants = 6
con_hists = []
for condition_code, group in by_condition:
    i = 0
    fig, axs = plt.subplots((len(question_names) + 1) // 2, 2, figsize=(8.5, 11))
    condition_name = str(condition_code)
    fig.suptitle("Condition " + condition_name)
    for question in question_names:
        row = math.floor(i / 2)
        column = i % 2
        ax = axs[row][column]
        ax.set_title(question)
        ax.set_ylim((0, num_participants))
        ax.set_yticks(np.arange(0, num_participants, 1))
        ax.grid(axis="y")
        q_data = group[question]
        # as_array = question.to_numpy(dtype=np.int).flatten()
        as_dict = q_data.append(pd.Series([1, 2, 3, 4, 5]), ignore_index=True).value_counts()
        as_dict = {key: value - 1 for key, value in as_dict.items()}
        if 6 in as_dict.keys():
            print("IMPUTATION PROBLEM")
            del as_dict[6]
        values = [value for _, value in sorted(as_dict.items())]
        ax.bar([1, 2, 3, 4, 5], values)
        i += 1
    plt.show()
    con_hists.append(fig)

# Scatterplots
feat_v_attr_scatters = []
for question in question_names:
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
        q_data = condition_ratings[question]
        feat_data = condition_ratings["features"].apply(lambda row: row[i])
        ax.plot(feat_data, q_data, "bo")
    plt.show()
    feat_v_attr_scatters.append(fig)

fig, axs = plt.subplots((len(question_names) + 1) // 2, 2, figsize=(8.5, 11))
condition_name = "All Conditions"
i = 0
for question in question_names:
    row = math.floor(i / 2)
    column = i % 2
    ax = axs[row][column]
    ax.set_title(question)
    ax.set_ylim((0, 55))
    ax.set_yticks(np.arange(0, 71, 10))
    ax.grid(axis="y")
    q_data = condition_ratings[question]
    # as_array = question.to_numpy(dtype=np.int).flatten()
    as_dict = q_data.append(pd.Series([1, 2, 3, 4, 5]), ignore_index=True).value_counts()
    as_dict = {key: value - 1 for key, value in as_dict.items()}
    if 6 in as_dict.keys():
        print("IMPUTATION PROBLEM")
        del as_dict[6]
    values = [value for _, value in sorted(as_dict.items())]
    ax.bar([1, 2, 3, 4, 5], values)
    i += 1
# plt.show()
con_hists.append(fig)



fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
condition_name = "All Conditions, All Questions"
ax.set_title(condition_name)

ax.grid(axis="y")
q_data = pd.Series(condition_ratings[question_names].to_numpy().flatten())
as_dict = q_data.append(pd.Series([1, 2, 3, 4, 5]), ignore_index=True).value_counts()
as_dict = {key: value - 1 for key, value in as_dict.items()}
if 6 in as_dict:
    del as_dict[6]
values = [value for _, value in sorted(as_dict.items())]
ax.set_ylim((0, max(values)))
ax.set_yticks(np.arange(0, max(values), 10))
ax.bar([1, 2, 3, 4, 5], values)
plt.show()
con_hists.append(fig)

with PdfPages('../data/plots.pdf') as pp:
    pp.savefig(fa_plot)
    pp.savefig(conf_plots)
    for fig in con_hists:
        pp.savefig(fig)
    for fig in feat_v_attr_scatters:
        pp.savefig(fig)
