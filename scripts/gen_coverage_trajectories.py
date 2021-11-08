"""
Takes an input pool of demonstrated trajectories and generates
additional samples to form a pool of trajectories with diverse
featurizations.
"""
import itertools
import json

from functools import partial
import numpy as np
import random
from scipy import spatial

from search.routine import astar
from search.coverage import trajectory_features, CoverageNode
from search.metric import coverage_manhattan
from search.sampling import sample_diverse, sample_perturb_neighbor_pool
from search.trajectory import TrajectoryNode
from search.util import load_map, shelve_it, pretty_plan

random.seed(0)


@shelve_it("trajectoriesa.shelf")
def get_coverage_plan(start, to_cover, grid_orig):
    return astar(CoverageNode(to_cover, start), CoverageNode([], start), grid_orig, coverage_manhattan)


if __name__ == '__main__':
    grid, bedroom = load_map("interface/assets/map32.tmx")

    plan = get_coverage_plan((10, 9), bedroom, grid)
    print(pretty_plan(plan))

    raw_plan = list(map(lambda x: x.point, plan))
    featurizer = partial(trajectory_features, bedroom, grid)
    orig_features = featurizer(raw_plan)
    print(orig_features)
    TrajectoryNode.featurizer = featurizer

    curious = [(10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9), (17, 9), (18, 9), (19, 9), (19, 8),
               (20, 8), (20, 7), (20, 8), (21, 8), (21, 9), (20, 9), (20, 10), (21, 10), (21, 11), (22, 11), (22, 10),
               (23, 10), (23, 11), (24, 11), (24, 10), (25, 10), (25, 9), (25, 8), (25, 7), (25, 8), (26, 8), (27, 8),
               (27, 9), (26, 9), (26, 10), (27, 10), (27, 11), (27, 12), (26, 12), (26, 11), (25, 11), (25, 12),
               (25, 13), (24, 13), (24, 12), (23, 12), (23, 13), (22, 13), (22, 12), (21, 12), (21, 13), (20, 13),
               (20, 12), (19, 12), (19, 11), (20, 11), (20, 10), (20, 9), (19, 9), (19, 10), (18, 10), (17, 10),
               (16, 10), (16, 10), (16, 10), (16, 10), (17, 10), (17, 9), (17, 9), (17, 9), (17, 9), (17, 9), (16, 9),
               (15, 9), (14, 9), (13, 9), (12, 9), (12, 9), (12, 9), (12, 9), (12, 9), (12, 8), (12, 7), (12, 6),
               (12, 6), (12, 6), (12, 6), (12, 6), (12, 6), (12, 6), (12, 6), (12, 6), (12, 6), (12, 6), (12, 6),
               (12, 6), (12, 6), (12, 6), (12, 6), (12, 6), (12, 7), (12, 8), (12, 9), (11, 9), (10, 9)]
    broken = [(10, 9), (11, 9), (11, 8), (12, 8), (12, 9), (13, 9), (13, 10), (13, 11), (14, 11), (14, 10), (15, 10),
              (16, 10), (16, 11), (17, 11), (17, 10), (17, 9), (18, 9), (19, 9), (19, 10), (18, 10), (18, 11), (19, 11),
              (20, 11), (21, 11), (21, 10), (21, 9), (20, 9), (20, 10), (21, 10), (21, 9), (20, 9), (20, 10), (21, 10),
              (21, 9), (20, 9), (20, 10), (21, 10), (21, 9), (21, 8), (20, 8), (20, 9), (21, 9), (21, 8), (20, 8),
              (20, 9), (21, 9), (21, 8), (20, 8), (20, 9), (19, 9), (19, 8), (19, 9), (19, 10), (18, 10), (17, 10),
              (17, 9), (16, 9), (16, 10), (17, 10), (18, 10), (18, 9), (17, 9), (16, 9), (15, 9), (15, 10), (14, 10),
              (13, 10), (13, 9), (12, 9), (12, 10), (13, 10), (13, 9), (13, 8), (12, 8), (11, 8), (10, 8), (10, 7),
              (10, 8), (10, 9)]
    energetic = [(11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9), (17, 9), (18, 9), (19, 9), (20, 9), (20, 8),
                 (21, 8), (21, 9), (20, 9), (19, 9), (19, 8), (19, 9), (19, 10), (19, 11), (20, 11), (20, 10), (19, 10),
                 (20, 10), (20, 9), (20, 8), (20, 7), (20, 8), (20, 9), (20, 10), (21, 10), (22, 10), (22, 11),
                 (22, 12), (22, 11), (22, 10), (21, 10), (20, 10), (20, 11), (20, 12), (19, 12), (19, 11), (20, 11),
                 (20, 12), (21, 12), (21, 13), (20, 13), (21, 13), (21, 12), (21, 11), (21, 10), (22, 10), (23, 10),
                 (24, 10), (25, 10), (25, 9), (25, 8), (25, 7), (25, 8), (26, 8), (27, 8), (27, 9), (26, 9), (26, 8),
                 (27, 8), (26, 8), (26, 9), (25, 9), (25, 10), (26, 10), (26, 9), (27, 9), (27, 10), (27, 11), (26, 11),
                 (26, 10), (25, 10), (25, 11), (26, 11), (27, 11), (27, 12), (26, 12), (26, 11), (25, 11), (25, 12),
                 (25, 13), (24, 13), (24, 12), (24, 11), (25, 11), (24, 11), (24, 10), (25, 10), (26, 10), (26, 9),
                 (27, 9), (27, 10), (26, 10), (25, 10), (25, 11), (24, 11), (24, 12), (23, 12), (23, 11), (22, 11),
                 (22, 10), (23, 10), (24, 10), (24, 11), (23, 11), (22, 11), (22, 12), (23, 12), (23, 11), (24, 11),
                 (24, 12), (23, 12), (24, 12), (24, 11), (24, 10), (25, 10), (24, 10), (24, 11), (24, 12), (23, 12),
                 (23, 11), (22, 11), (23, 11), (23, 12), (23, 13), (22, 13), (22, 12), (21, 12), (21, 11), (21, 10),
                 (20, 10), (19, 10), (19, 9), (20, 9), (21, 9), (21, 10), (21, 11), (20, 11), (19, 11), (18, 11),
                 (17, 11), (16, 11), (15, 11), (14, 11), (13, 11), (12, 11), (11, 11), (10, 11), (10, 10), (10, 9)]
    lazy = [(10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9), (17, 9), (18, 9), (19, 9), (19, 8), (20, 8),
            (20, 7), (20, 8), (21, 8), (21, 9), (20, 9), (20, 10), (21, 10), (21, 11), (21, 12), (21, 13), (20, 13),
            (20, 12), (20, 11), (19, 11), (19, 12), (19, 11), (19, 10), (18, 10), (17, 10), (16, 10), (15, 10),
            (14, 10), (13, 10), (12, 10), (11, 10), (10, 10), (10, 9)]
    lost = [(11, 9), (12, 9), (13, 9), (13, 9), (13, 9), (13, 10), (13, 11), (13, 11), (13, 11), (12, 11), (12, 11),
            (12, 11), (12, 11), (12, 11), (12, 10), (12, 9), (12, 8), (12, 8), (12, 8), (12, 8), (13, 8), (14, 8),
            (14, 8), (14, 8), (14, 9), (14, 10), (14, 10), (14, 10), (14, 10), (14, 10), (14, 10), (14, 10), (14, 10),
            (13, 10), (12, 10), (11, 10), (10, 10), (10, 10), (10, 9)]
    scared = [(10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9), (17, 9), (17, 9), (17, 9), (17, 9),
              (17, 9), (18, 9), (18, 10), (18, 11), (17, 11), (16, 11), (15, 11), (15, 10), (16, 10), (17, 10),
              (17, 10), (17, 10), (17, 10), (17, 10), (17, 10), (17, 10), (17, 10), (17, 10), (17, 10), (17, 11),
              (18, 11), (18, 11), (18, 11), (17, 11), (16, 11), (15, 11), (14, 11), (13, 11), (12, 11), (11, 11),
              (10, 11), (10, 10), (10, 9)]

    pool = [raw_plan, curious, broken, energetic, lazy, lost, scared]
    feat_pool = list(map(TrajectoryNode.featurizer, pool))
    feat_pool = np.array(feat_pool, dtype=np.float)

    print("Initial pool")
    print("Mean manhat diff: {}".format(spatial.distance.pdist(feat_pool, "cityblock").mean()))
    print(list(pool))

    print("Diverse")
    div_trajs, div_feats = sample_diverse(pool, 0, grid)
    # print("Mean manhat diff: {}".format(spatial.distance.pdist(div_feats, "cityblock").mean()))

    print("Neighbors")
    # nei_trajs, nei_feats = sample_neighbors(pool, 2, grid)
    nei_trajs = list(itertools.chain.from_iterable(sample_perturb_neighbor_pool(pool, 2, grid, featurizer)))
    nei_feats = np.array(list(map(TrajectoryNode.featurizer, nei_trajs)))
    print("Mean manhat diff: {}".format(spatial.distance.pdist(nei_feats, "cityblock").mean()))

    pool = pool + nei_trajs
    data = {"trajectories": str(pool), "features": np.vstack(map(TrajectoryNode.featurizer, pool)).tolist()}
    with open("initial_trajs.json") as f:
        json.dump(data, f)
