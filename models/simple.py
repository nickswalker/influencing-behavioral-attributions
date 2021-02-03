import numpy as np

from sklearn import linear_model

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC


def fit_lin_reg(condition_ratings, variable_names):
    print("Linear regression targeting items -------------------")
    scaler = StandardScaler()
    scaler.fit(condition_ratings[variable_names].to_numpy())

    scaled = scaler.transform(condition_ratings[variable_names])

    y = scaled
    x = np.vstack(condition_ratings["features"])

    model = linear_model.LinearRegression()
    model.fit(x, y)
    # print(model.coef_)
    # print(model.intercept_)
    print("lin reg R2 total\t\t {:.2}".format(model.score(x, y)))

    for q_name in variable_names:
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


def fit_log_reg(x,y):
    model = linear_model.LogisticRegressionCV(max_iter=3000)
    model.fit(x, y)
    return model


def bin_likert(data):
    data[data < 3] = 0
    data[data == 3] = 1
    data[data > 3] = 2
    return data


def bin_factor_score(data):
    low = data < -.5
    mid = (-.5 < data) & (data < .5)
    high = .5 < data
    data[low] = 0
    data[mid] = 1
    data[high] = 2
    return data.astype(int)


def fit_mlp(x, y):
    clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=0, solver="lbfgs",hidden_layer_sizes=[20,20], activation="relu", max_iter=3000))
    clf.fit(x, y)
    return clf

def fit_mlp_regressor(x, y):
    clf = make_pipeline(StandardScaler(), MLPRegressor(random_state=0, solver="lbfgs", hidden_layer_sizes=[15,15], activation="relu", max_iter=3000))
    clf.fit(x, y)
    return clf


def fit_svm(x, y):
    clf = make_pipeline(StandardScaler(), SVC(random_state=0, probability=False, class_weight="balanced"))
    clf.fit(x, y)
    return clf

def fit_knn(x,y):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x, y)
    return neigh