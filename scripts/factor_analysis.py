from sklearn.impute import IterativeImputer

from processing.loading import process_turk_files
import numpy as np
import pandas as pd
from joblib import dump
from factor_analyzer import FactorAnalyzer

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