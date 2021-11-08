# We have to enable this because it's experimental
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from processing.loading import process_turk_files
import numpy as np
import pandas as pd
from joblib import dump
from factor_analyzer import FactorAnalyzer
from processing.mappings import old_question_names as all_question_names

condition_ratings = []
demos = []
other_data = []
max_id = 0
for base in ["pilot1", "active1", "active2", "mdn_active1", "mdn_active2"]:
    cr, d, o, _ = process_turk_files("data/" + base + ".csv", traj_file="data/" + base + "_trajs.json")
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
first_step_data = condition_ratings[0]
first_q_names = [q_name for q_name in all_question_names if q_name in first_step_data.columns]

transformer = FactorAnalyzer(n_factors=3, rotation="promax", )
analysis = transformer.fit(first_step_data[first_q_names].to_numpy())

# Analysis across three SVM pilot factor
transformer2_3factor = FactorAnalyzer(n_factors=3, rotation="promax")
analysis23 = transformer2_3factor.fit(pd.concat(condition_ratings[:3])[first_q_names].to_numpy())

transformer2_4factor = FactorAnalyzer(n_factors=4, rotation="promax")
analysis24 = transformer2_4factor.fit(pd.concat(condition_ratings[:3])[first_q_names].to_numpy())

transformer2_5factor = FactorAnalyzer(n_factors=5, rotation="promax")
analysis25 = transformer2_5factor.fit(pd.concat(condition_ratings[:3])[first_q_names].to_numpy())

# It became apparent that this item wasn't working
small_question_names = first_q_names.copy()
small_question_names.remove("lazy")

transformer4 = FactorAnalyzer(n_factors=3, rotation="promax")
analysis4 = transformer4.fit(pd.concat(condition_ratings[:3])[small_question_names].to_numpy())

# And finally that this item too was weak
small_question_names.remove("energetic")

transformer5_3factor = FactorAnalyzer(n_factors=3, rotation="promax")
analysis53 = transformer5_3factor.fit(pd.concat(condition_ratings[:3])[small_question_names].to_numpy())

transformer5_4factor = FactorAnalyzer(n_factors=4, rotation="promax")
analysis54 = transformer5_4factor.fit(pd.concat(condition_ratings[:3])[small_question_names].to_numpy())

transformer5_5factor = FactorAnalyzer(n_factors=5, rotation="promax")
analysis55 = transformer5_5factor.fit(pd.concat(condition_ratings[:3])[small_question_names].to_numpy())

# These analyses came too late to  make it into our scale design
# With new data (inquisitive added)
transformer6 = FactorAnalyzer(n_factors=4, rotation="promax")
analysis6 = transformer6.fit(pd.concat(condition_ratings[3:])[all_question_names].to_numpy())

# We retrace some steps of dropping items with this newer data
# New data, lazy dropped
new_small_question_names = all_question_names.copy()
new_small_question_names.remove("lazy")
transformer7 = FactorAnalyzer(n_factors=3, rotation="promax")
analysis7 = transformer7.fit(pd.concat(condition_ratings[3:])[new_small_question_names].to_numpy())

new_small_question_names.remove("energetic")

transformer8 = FactorAnalyzer(n_factors=3, rotation="promax")
analysis8 = transformer8.fit(pd.concat(condition_ratings[3:])[new_small_question_names].to_numpy())

chosen_transformer = transformer5_3factor
question_names = small_question_names

print(np.around(chosen_transformer.loadings_, 2))
#dump(chosen_transformer, "factor_model.pickle")