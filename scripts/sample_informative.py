import glob
import itertools
import json
import random
from functools import partial

from joblib import load
import pandas as pd

import numpy as np
from scipy.spatial.distance import cdist

from models.util import gen_samples, process_turk_files, load_trajectory_pool, attributions, load_demo_pool
from search.coverage import trajectory_features
from search.sampling import sample_neighbors, sample_neighbors_for_pool, sample_random_goal_neighbor
from search.trajectory import TrajectoryNode
from search.util import load_map

random.seed(0)
num_conditions = 21

num_attributions = len(attributions)


def iter_uncertainty_sample(model, batch, batch_feats, population, pop_feats, used_indices, balance_lambda):
    plane_dist = model.decision_function(pop_feats)
    # Kludge to make sure we don't reuse these
    total_plane_dist = abs(plane_dist).sum(axis=1)
    if len(batch) == 0:
      print(total_plane_dist.argmin())
      return total_plane_dist.argmin(), population[total_plane_dist.argmin()]
    total_plane_dist[np.array(used_indices)] = float("inf")
    # Get a matrix of every pairwise similarity
    # Note: cdist is a _distance_, so we'll flip it to similarity (1 means equal, 0 means orthogonal) since that's the usual notation
    similarity = 1 - cdist(pop_feats, batch_feats, "cosine")
    # We're gonna want to add the sample that's highly dissimilar to everything in the batch already
    peak_similarity = similarity.max(axis=1)
    scores = balance_lambda * total_plane_dist + (1 - balance_lambda) * peak_similarity
    print(scores.argmin())
    return scores.argmin(), population[scores.argmin()]


if __name__ == '__main__':
    grid, bedroom = load_map("interface/assets/map32.tmx")

    condition_ratings, demo_data, other_data = process_turk_files(glob.glob("data/pilot1.csv"))
    prompts = list(demo_data["aid"])
    demo_trajs = list(demo_data["traj"])

    featurizer = partial(trajectory_features, bedroom, grid)
    TrajectoryNode.featurizer = featurizer

    factor_clfs = load("data/factor_svms.pickle")

    neighbors = sample_neighbors_for_pool(demo_trajs, 3, grid)
    #neighbors = sample_random_goal_neighbor(demo_trajs, 3, grid)
    population = demo_trajs + list(itertools.chain.from_iterable(neighbors))
    population_features = list(map(featurizer, population))
    origin_per_traj = list(range(len(demo_trajs)))
    for i, neighbors in enumerate(neighbors):
        origin_per_traj = origin_per_traj + [i] * len(neighbors)

    per_factor = []
    used_indices = []
    for clf in factor_clfs:
        per_factor.append([])
        for _ in range(7):
            index, sample = iter_uncertainty_sample(clf, per_factor[-1], list(map(featurizer, per_factor[-1])), population, population_features, used_indices, .7)
            per_factor[-1].append(sample)
            used_indices.append(index)

    print("Len demos {}, used {}".format(len(demo_trajs), int((pd.DataFrame(used_indices) < len(demo_trajs)).sum())))
    pool_trajs = []
    for i in used_indices:
         print("{} ({}, {})".format(i, origin_per_traj[i], prompts[origin_per_traj[i]]))
         pool_trajs.append(population[i])
    pool = {"trajectories": str(pool_trajs), "features": list(map(featurizer, pool_trajs))}
    with open("data/sampled_pool.json", "w") as f:
        json.dump(pool, f)