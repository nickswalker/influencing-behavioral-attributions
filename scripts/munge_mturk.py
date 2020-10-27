import glob
import math
import re

import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis, PCA
from factor_analyzer import FactorAnalyzer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split

from matplotlib.backends.backend_pdf import PdfPages

question_code_to_name = {"b1": "brave", "b2": "broken", "b3": "clumsy", "b4": "curious", "b5": "efficient",
                         "b6": "energetic", "b7": "intelligent", "b8": "lazy", "b9": "reliable", "b10": "scared"}
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
feats = [[1.0, 0.2911392405063291, 0.5602836879432624, 0.6363636363636364, 0.07792207792207792, 0.2978723404255319, 0.0,
          0.23404255319148937, 0.09507640067911714, 0.0, 0.02564102564102564],
         [0.0, 0.06060606060606055, 0.23404255319148937, 0.5806451612903226, 0.0967741935483871, 0.0, 0.0, 0.0,
          0.05263157894736842, 0.0, 0.0625],
         [1.0, 0.2784810126582279, 0.5602836879432624, 0.6493506493506493, 0.11688311688311688, 0.2978723404255319,
          0.02127659574468085, 0.23404255319148937, 0.0967741935483871, 0.0, 0.02564102564102564],
         [1.0, 0.1428571428571429, 0.5460992907801419, 0.48, 0.16, 0.3404255319148936, 0.0, 0.2127659574468085,
          0.11205432937181664, 0.0, 0.02631578947368421],
         [0.9574468085106383, 0.4563106796116505, 0.7304964539007093, 0.33663366336633666, 0.7722772277227723,
          0.2978723404255319, 0.0, 0.23404255319148937, 0.09507640067911714, 0.0, 0.0196078431372549],
         [0.25531914893617025, 0.21621621621621623, 0.2624113475177305, 0.7142857142857143, 0.0, 0.0851063829787234,
          0.0, 0.0851063829787234, 0.04923599320882852, 0.0, 0.05555555555555555],
         [0.9787234042553191, 0.2857142857142857, 0.5460992907801419, 0.7066666666666667, 0.12, 0.2978723404255319, 0.0,
          0.23404255319148937, 0.0933786078098472, 0.0, 0.02631578947368421],
         [1.0, 0.504424778761062, 0.8014184397163121, 0.5045045045045045, 0.16216216216216217, 0.2978723404255319, 0.0,
          0.8723404255319149, 0.09507640067911714, 0.0, 0.017857142857142856],
         [0.9787234042553191, 0.42708333333333337, 0.6808510638297872, 0.2872340425531915, 0.06382978723404255,
          0.2978723404255319, 0.0, 0.3191489361702128, 0.0933786078098472, 0.24210526315789474, 0.5052631578947369],
         [0.9574468085106383, 0.3296703296703297, 0.6453900709219859, 0.6067415730337079, 0.10112359550561797,
          0.3404255319148936, 0.0425531914893617, 0.425531914893617, 0.1035653650254669, 0.0, 0.022222222222222223],
         [0.9787234042553191, 0.30379746835443033, 0.5602836879432624, 0.6363636363636364, 0.0, 0.2978723404255319, 0.0,
          0.23404255319148937, 0.0933786078098472, 0.0, 0.02564102564102564],
         [0.9361702127659575, 0.3205128205128205, 0.5531914893617021, 0.7368421052631579, 0.039473684210526314,
          0.2553191489361702, 0.0, 0.2553191489361702, 0.0899830220713073, 0.012987012987012988, 0.025974025974025976],
         [0.0, 0.05714285714285716, 0.24822695035460993, 0.48484848484848486, 0.09090909090909091, 0.0, 0.0, 0.0,
          0.05602716468590832, 0.0, 0.058823529411764705],
         [0.0, 0.28888888888888886, 0.3191489361702128, 0.5581395348837209, 0.06976744186046512, 0.0,
          0.2978723404255319, 0.0, 0.05432937181663837, 0.18181818181818182, 0.09090909090909091],
         [0.0, 0.06060606060606055, 0.23404255319148937, 0.5806451612903226, 0.0, 0.0, 0.0, 0.0, 0.05263157894736842,
          0.0, 0.0625],
         [0.7021276595744681, 0.47619047619047616, 0.5957446808510638, 0.5975609756097561, 0.07317073170731707,
          0.425531914893617, 0.0, 0.3191489361702128, 0.07470288624787776, 0.012048192771084338, 0.04819277108433735],
         [0.9574468085106383, 0.2921348314606742, 0.6312056737588653, 0.4482758620689655, 0.10344827586206896,
          0.3404255319148936, 0.02127659574468085, 0.23404255319148937, 0.10696095076400679, 0.0, 0.022727272727272728],
         [1.0, 0.2784810126582279, 0.5602836879432624, 0.7402597402597403, 0.11688311688311688, 0.2978723404255319,
          0.02127659574468085, 0.2553191489361702, 0.0967741935483871, 0.0, 0.02564102564102564],
         [1.0, 0.15384615384615385, 0.6453900709219859, 0.48314606741573035, 0.16853932584269662, 0.3404255319148936,
          0.0, 0.2127659574468085, 0.1307300509337861, 0.0, 0.022222222222222223],
         [1.0, 0.16190476190476188, 0.7446808510638298, 0.46601941747572817, 0.17475728155339806, 0.3404255319148936,
          0.0, 0.2127659574468085, 0.1494057724957555, 0.0, 0.019230769230769232],
         [1.0, 0.18487394957983194, 0.8439716312056738, 0.47863247863247865, 0.15384615384615385, 0.3404255319148936,
          0.0, 0.2127659574468085, 0.16468590831918506, 0.0, 0.01694915254237288],
         [1.0, 0.45871559633027525, 0.7730496453900709, 0.3177570093457944, 0.7289719626168224, 0.2765957446808511, 0.0,
          0.2553191489361702, 0.100169779286927, 0.0, 0.018518518518518517],
         [0.9574468085106383, 0.4363636363636364, 0.7801418439716312, 0.32407407407407407, 0.75, 0.48936170212765956,
          0.0, 0.23404255319148937, 0.10526315789473684, 0.009174311926605505, 0.03669724770642202],
         [0.9574468085106383, 0.4473684210526315, 0.8085106382978723, 0.3482142857142857, 0.6964285714285714,
          0.3191489361702128, 0.19148936170212766, 0.2553191489361702, 0.10696095076400679, 0.061946902654867256,
          0.035398230088495575],
         [0.25531914893617025, 0.2564102564102564, 0.2765957446808511, 0.7027027027027027, 0.0, 0.0851063829787234, 0.0,
          0.10638297872340426, 0.04923599320882852, 0.05263157894736842, 0.15789473684210525],
         [0.276595744680851, 0.26415094339622647, 0.375886524822695, 0.6666666666666666, 0.0, 0.0851063829787234,
          0.2978723404255319, 0.0851063829787234, 0.06621392190152801, 0.11538461538461539, 0.07692307692307693],
         [0.25531914893617025, 0.23076923076923073, 0.46099290780141844, 0.7142857142857143, 0.0, 0.0851063829787234,
          0.0, 0.0851063829787234, 0.08488964346349745, 0.0, 0.03125],
         [0.6808510638297872, 0.4878048780487805, 0.5815602836879432, 0.575, 0.1125, 0.2978723404255319, 0.0,
          0.3829787234042553, 0.07130730050933787, 0.012345679012345678, 0.04938271604938271],
         [0.9148936170212766, 0.28, 0.5319148936170213, 0.684931506849315, 0.1232876712328767, 0.19148936170212766,
          0.02127659574468085, 0.23404255319148937, 0.09168081494057725, 0.0, 0.02702702702702703],
         [0.9574468085106383, 0.3214285714285714, 0.5957446808510638, 0.6585365853658537, 0.21951219512195122,
          0.3191489361702128, 0.0, 0.2553191489361702, 0.0967741935483871, 0.012048192771084338, 0.04819277108433735],
         [1.0, 0.504424778761062, 0.8014184397163121, 0.5045045045045045, 0.16216216216216217, 0.2978723404255319, 0.0,
          0.8723404255319149, 0.09507640067911714, 0.0, 0.017857142857142856],
         [0.8085106382978724, 0.5480769230769231, 0.7375886524822695, 0.47058823529411764, 0.17647058823529413,
          0.3617021276595745, 0.0, 0.8085106382978723, 0.07979626485568761, 0.009708737864077669, 0.038834951456310676],
         [0.9787234042553191, 0.5080645161290323, 0.8794326241134752, 0.30327868852459017, 0.19672131147540983,
          0.3191489361702128, 0.0, 0.851063829787234, 0.1035653650254669, 0.008130081300813009, 0.032520325203252036],
         [0.7234042553191489, 0.3157894736842105, 0.5390070921985816, 0.4864864864864865, 0.04054054054054054,
          0.2765957446808511, 0.0, 0.2765957446808511, 0.08828522920203735, 0.14666666666666667, 0.32],
         [0.9574468085106383, 0.4329896907216495, 0.6879432624113475, 0.2631578947368421, 0.15789473684210525,
          0.2765957446808511, 0.02127659574468085, 0.3191489361702128, 0.0933786078098472, 0.22916666666666666,
          0.4791666666666667],
         [0.9787234042553191, 0.4363636363636364, 0.7801418439716312, 0.2777777777777778, 0.05555555555555555,
          0.2978723404255319, 0.0, 0.3191489361702128, 0.10526315789473684, 0.24770642201834864, 0.5137614678899083]]
feats = np.stack(feats)


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
    factor_names = ["factor"+str(i) for i in range(4)]
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
    factor_names = ["factor" + str(i) for i in range(4)]
    y = condition_ratings[factor_names].copy()
    x = np.vstack(condition_ratings["features"])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
    for factor in factor_names:
        f_data = y_train[factor].copy()
        f_data[f_data < f_data.quantile(.33)] = 0
        f_data[(f_data.quantile(.33) < f_data) & (f_data < f_data.quantile(.66))] = 1
        f_data[f_data.quantile(.66) < f_data] = 2
        f_data = f_data.astype(int)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, f_data.to_numpy())
        t_data = y_test[factor].copy()
        t_data[t_data < f_data.quantile(.33)] = 0
        t_data[(f_data.quantile(.33) < t_data) & (t_data < t_data.quantile(.66))] = 1
        t_data[f_data.quantile(.66) < t_data] = 2
        t_data = t_data.astype(int)
        print("svm acc {} \t\t {:.2}".format(factor, clf.score(X_test, t_data.to_numpy())))


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
        columns = ["Input.id" + str(n), "WorkerId"] + answers_cols
        condition_views.append(frame[columns].rename(columns=drop_con_number).rename(columns=drop_id_trailing_number))
    condition_ratings = pd.concat(condition_views, ignore_index=True)
    # We loaded many different files. Some may not have as many command columns as others
    condition_ratings.dropna(inplace=True)

    # Drop all the extra input or answer fields except those you want to use for additional stats
    other_data = frame.drop(
        columns=[c for c in frame.columns if
                 ("Input" in c or "Answer" in c) and c != "Answer.comment" and c != "Input.hitid"])

    nice_names = {"Answer.paraphrase": "paraphrase", "Input.command": "command", "Answer.newcommand": "command",
                  "Input.hitid": "hitid"}
    other_data.rename(columns=nice_names, inplace=True)
    # Weird behavior on explain column if I do "Answer."
    condition_ratings.columns = condition_ratings.columns.str.lstrip(r'Answer')
    condition_ratings.columns = condition_ratings.columns.str.lstrip(r'.')
    condition_ratings.rename(columns=question_code_to_name, inplace=True)
    condition_ratings.columns = condition_ratings.columns.str.lstrip("Input.")
    return condition_ratings, other_data


condition_ratings, other_data = process_turk_files(glob.glob("Batch_*.csv"))

# Add feats and traj data
condition_ratings["features"] = condition_ratings["id"].apply(lambda x: feats[x]).to_numpy()
condition_ratings["trajectories"] = condition_ratings["id"].apply(lambda x: feats[x]).to_numpy()

# Fill in missing values
imp = IterativeImputer(missing_values=6, max_iter=10, random_state=0)
imp.fit(condition_ratings[question_names].to_numpy())
condition_ratings[question_names] = np.rint(imp.transform(condition_ratings[question_names].to_numpy())).astype(int)

# Create factor loadings
num_factors = 4
transformer = FactorAnalyzer(num_factors)
analysis = transformer.fit(condition_ratings[question_names].to_numpy())
fa_plot = make_fa_plots(condition_ratings, analysis)
loadings = pd.DataFrame(analysis.loadings_.T,columns=question_names)

for i in range(num_factors):
    condition_ratings["factor"+str(i)] = (condition_ratings[question_names] * loadings.iloc[i]).sum(axis=1)

fit_lin_reg(condition_ratings)
fit_classification(condition_ratings)
by_condition = condition_ratings.groupby("id")
cond_qual = []
for i in range(36):
    cond_qual.append([])
for name, group in by_condition:
    print(name)
    for i, (describe, explain) in group[["describe", "explain"]].iterrows():
        print(describe)
        print(explain)
        print("")
        cond_qual[int(name)].append(describe + "---" + explain)

print(cond_qual)
turker_performance = pd.DataFrame()
turker_performance["HITTime"] = other_data.groupby("WorkerId")["WorkTimeInSeconds"].mean()
# turker_performance["Comment"] = other_data.groupby("WorkerId")["Answer.comment"].apply(lambda x: ','.join(x))

turker_performance.to_csv("turker_stats.txt", index=False)

# Histograms
num_participants = 6
con_hists = []
for condition_code, group in by_condition:
    i = 0
    fig, axs = plt.subplots(5, 2, figsize=(8.5, 11))
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

fig, axs = plt.subplots(5, 2, figsize=(8.5, 11))
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
    values = [value for _, value in sorted(as_dict.items())]
    ax.bar([1, 2, 3, 4, 5], values)
    i += 1
# plt.show()
# con_hists.append(fig)



fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
condition_name = "All Conditions, All Questions"
ax.set_title(condition_name)

ax.grid(axis="y")
q_data = pd.Series(condition_ratings[question_names].to_numpy().flatten())
as_dict = q_data.append(pd.Series([1, 2, 3, 4, 5]), ignore_index=True).value_counts()
as_dict = {key: value - 1 for key, value in as_dict.items()}
values = [value for _, value in sorted(as_dict.items())]
ax.set_ylim((0, max(values)))
ax.set_yticks(np.arange(0, max(values), 10))
ax.bar([1, 2, 3, 4, 5], values)
plt.show()
con_hists.append(fig)

with PdfPages('../data/{}_plots.pdf'.format(str(0))) as pp:
    pp.savefig(fa_plot)
    for fig in con_hists:
        pp.savefig(fig)
    for fig in feat_v_attr_scatters:
        pp.savefig(fig)
