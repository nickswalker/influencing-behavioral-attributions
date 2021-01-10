import ast
import random
import re

import pandas as pd
import json


def gen_samples(population, n, per_hit):
    pool = list(population)
    for _ in range(n):
        if per_hit > len(pool):
            sample = list(pool)
            pool = list(population)
            while True:
                extra = set(random.sample(pool, per_hit - len(sample)))
                if extra.isdisjoint(sample):
                    sample += list(extra)
                    pool = list(set(pool).difference(extra))
                    break
        else:
            sample = random.sample(pool, per_hit)
            pool = list(set(pool).difference(sample))
        random.shuffle(sample)
        yield sample


def load_trajectory_pool(file_name):
    with open(file_name) as f:
        data = json.load(f)

    import numpy as np
    return ast.literal_eval(data["trajectories"]), np.stack(data["features"])


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

    demos["Answer.traj"] = demos["Answer.traj"].apply(lambda x: ast.literal_eval(x))
    # Weird behavior on explain column if I do "Answer."
    condition_ratings.columns = condition_ratings.columns.str.replace(r'Answer.', "", regex=False)
    condition_ratings.rename(columns=question_code_to_name, inplace=True)
    condition_ratings.columns = condition_ratings.columns.str.replace("Input.", "", regex=False)

    demos.columns = demos.columns.str.replace("Answer.", "", regex=False)
    demos.columns = demos.columns.str.replace("Input.", "", regex=False)
    return condition_ratings, demos, other_data


question_code_to_name = {"b1": "broken", "b2": "clumsy", "b3": "competent", "b4": "confused", "b5": "curious",
                         "b6": "efficient", "b7": "energetic", "b8": "focused", "b9": "intelligent",
                         "b10": "investigative", "b11": "lazy", "b12": "lost", "b13": "reliable", "b14": "responsible"}
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
attributions = ["broken", "clumsy", "competent", "confused", "curious", "efficient", "energetic", "focused", "intelligent", "investigative", "lazy", "lost", "reliable", "responsible"]