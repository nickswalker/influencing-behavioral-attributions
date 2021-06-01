import glob
import itertools
import json
import os
import random
import sys
from functools import partial

import torch
import pandas as pd

import numpy as np
from scipy.spatial.distance import cdist

from models.mdn import ens_uncertainty_kl
from models.nn import load_ensemble
from models.util import gen_samples
from processing.loading import load_trajectory_pool, load_demo_pool, process_turk_files
from processing.mappings import attributions
from search.coverage import trajectory_features
from search.sampling import sample_perturb_neighbor_pool
from search.trajectory import TrajectoryNode
from search.util import load_map


random.seed(0)
num_conditions = 21

num_attributions = len(attributions)


def iter_uncertainty_sample(unc, batch, pop_feats, used_indices, balance_lambda):
    if len(batch) == 0:
      return unc.argmax()
    unc[np.array(used_indices)] = float("-inf")
    # Get a matrix of every pairwise similarity
    # Note: cdist is a _distance_, so we'll flip it to similarity (1 means equal, 0 means orthogonal) since that's the usual notation
    similarity = 1 - cdist(pop_feats, pop_feats[np.array(batch)], "cosine")
    # We care about max similarity to something in the batch already
    similarity = similarity.max(1)
    # We're gonna want to add the sample that's highly dissimilar to everything in the batch already (low sim value) and
    # the model is uncertain about (high unc value). We flip unc sign to make this work.
    scores = balance_lambda * -unc + (1 - balance_lambda) * similarity
    return scores.argmin()


if __name__ == '__main__':
    grid, bedroom = load_map("interface/assets/map32.tmx")

    condition_ratings, demo_data, other_data = process_turk_files(["data/pilot1.csv","data/active1.csv", "data/active2.csv", "data/mdn_active1.csv"])
    prompts = list(demo_data["aid"])
    demo_trajs = list(demo_data["traj"])
    print(f"processing {len(demo_trajs)}")
    featurizer = partial(trajectory_features, bedroom, grid)
    TrajectoryNode.featurizer = featurizer

    ens = load_ensemble(os.getcwd() + "/data/ens2")

    # neighbors = sample_neighbors_for_pool(demo_trajs, 3, grid, featurizer)
    neighbors = sample_perturb_neighbor_pool(demo_trajs, 3, grid, featurizer)

    population = demo_trajs + list(itertools.chain.from_iterable(neighbors))
    population_features = list(map(featurizer, population))
    pool = {"trajectories": str(population), "features": list(map(featurizer, population))}
    with open("data/total_pool.json", "w") as f:
        json.dump(pool, f)

    origin_per_traj = list(range(len(demo_trajs)))
    for i, neighbors in enumerate(neighbors):
        origin_per_traj = origin_per_traj + [i] * len(neighbors)

    pop_feats = np.array(population_features)
    unc = ens_uncertainty_kl(ens, torch.Tensor(pop_feats)).detach().numpy()

    per_factor = []
    used_indices = []
    # Sort factors by neg mean unc (low value to high value), so we allocate
    # samples to most uncertain dimension first
    d_i = np.argsort(-unc.mean(0))
    print(d_i)
    for i in d_i:
        per_factor.append([])
        for _ in range(num_conditions // len(d_i)):
            # Indexing with list keeps singleton dim
            index = iter_uncertainty_sample(unc[:,i], per_factor[-1], pop_feats, used_indices, .7)
            per_factor[-1].append(index)
            used_indices.append(index)

    print("Len demos {}, used {}".format(len(demo_trajs), int((pd.DataFrame(used_indices) < len(demo_trajs)).sum())))
    pool_trajs = []
    for i in used_indices:
         print("{} ({}, {})".format(i, origin_per_traj[i], prompts[origin_per_traj[i]]))
         pool_trajs.append(population[i])
    pool = {"trajectories": str(pool_trajs), "features": list(map(featurizer, pool_trajs))}
    with open("data/sampled_pool.json", "w") as f:
        json.dump(pool, f)