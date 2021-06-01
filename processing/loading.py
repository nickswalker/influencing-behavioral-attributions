import ast
import json
import re

import pandas as pd

from processing.mappings import question_code_to_name


def load_trajectory_pool(file_name):
    with open(file_name) as f:
        data = json.load(f)
    if "features" in data.keys():
        import numpy as np
        return ast.literal_eval(data["trajectories"]), np.stack(data["features"])
    else:
        return ast.literal_eval(data["trajectories"]), None


def load_demo_pool(file_names):
    if not isinstance(file_names, list):
        file_names = [file_names]
    trajectories = []
    features = []
    prompts = []
    for file_name in file_names:
        with open(file_name) as f:
            data = json.load(f)
            trajectories += ast.literal_eval(data["trajectories"])
            prompts += data["prompts"]
    return trajectories, prompts


def process_turk_files(paths, filter_rejected=True, traj_file=None):
    if not isinstance(paths, list):
        paths = [paths]
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
    # We mark the example questions as condition 9999 to make them easy to ignore
    frame.drop(frame.columns[frame.columns.str.contains(r"9999")], axis=1, inplace=True)
    submitted = len(frame[frame["AssignmentStatus"] == "Submitted"])
    if submitted > 0:
        print("{} still pending".format(submitted))
    frame.drop(frame.columns[frame.columns.str.contains(r"noop")], axis=1, inplace=True)
    grouped = frame.groupby(frame.columns.str.extract(r"Answer.con_(\d+)_.*", expand=False), axis=1)

    comparison = pd.DataFrame()
    if "Answer.most_explain" in frame.columns:
        comparison = pd.DataFrame()
        comparison[["Input.id0", "Input.id1", "Input.id2", "Input.id3"]] = frame[
            ["Input.id0", "Input.id1", "Input.id2", "Input.id3"]]
        most = list(frame.columns[frame.columns.str.contains("most")])
        most.remove("Answer.most_explain")
        comparison["most_id"] = ((frame[most] == 1).idxmax(axis=1).str.extract(r"(\d+)$").astype(int) - 1)[0]
        def mapback(x):
            return x["Input.id"+str(x["most_id"])]
        comparison["most_id"] = comparison.apply(mapback, axis=1)
        least = list(frame.columns[frame.columns.str.contains("least")])
        least.remove("Answer.least_explain")
        comparison["least_id"] = ((frame[least] == 1).idxmax(axis=1).str.extract(r"(\d+)$").astype(int) - 1)[0]
        def mapback(x):
            return x["Input.id" + str(x["least_id"])]
        comparison["least_id"] = comparison.apply(mapback, axis=1)
        comparison["most_explain"] = frame["Answer.most_explain"]
        comparison["least_explain"] = frame["Answer.least_explain"]
        comparison.drop(["Input.id0", "Input.id1", "Input.id2", "Input.id3"], axis=1, inplace=True)

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
        # Input.id is the actual trajectory identifier, so this is where the information about ordering gets baked out
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
    demos = pd.DataFrame()
    if len(demo_views) > 0:
        demos = pd.concat(demo_views, ignore_index=True)
        # We loaded many different files. Some may not have as many command columns as others
        demos.dropna(inplace=True)
        demos["Answer.traj"] = demos["Answer.traj"].apply(lambda x: ast.literal_eval(x))

        demos.columns = demos.columns.str.replace("Answer.", "", regex=False)
        demos.columns = demos.columns.str.replace("Input.", "", regex=False)

    # Drop all the extra input or answer fields except those you want to use for additional stats
    other_data = frame.drop(
        columns=[c for c in frame.columns if
                 ("Input" in c or "Answer" in c) and c not in ["Answer.comment","Input.hitid", "Answer.gender", "Answer.ownership", "Answer.age"]])
    other_data["Answer.gender"] = other_data["Answer.gender"].str.lower().str.strip()
    other_data["Answer.ownership"] = other_data["Answer.ownership"].str.lower().str.strip()

    # Weird behavior on explain column if I do "Answer."
    condition_ratings.columns = condition_ratings.columns.str.replace(r'Answer.', "", regex=False)
    condition_ratings.rename(columns=question_code_to_name, inplace=True)
    condition_ratings.columns = condition_ratings.columns.str.replace("Input.", "", regex=False)

    if traj_file:
        trajectories, features = load_trajectory_pool(traj_file)
        # Add feats and traj data
        if features is not None:
            condition_ratings["features"] = condition_ratings["id"].apply(lambda x: features[x]).to_numpy()
        condition_ratings["trajectories"] = condition_ratings["id"].apply(lambda x: trajectories[x])

    return condition_ratings, demos, other_data, comparison