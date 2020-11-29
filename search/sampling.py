import random

import numpy as np

from search.trajectory import TrajectoryNode, ANY_PLAN
from search.metric import make_directed_l2, feat_l2
from search.routine import hill_climb


def sample_diverse(pool, samples_per_orig, grid):
    feat_pool = list(map(TrajectoryNode.featurizer, pool))
    feat_pool = np.array(feat_pool, dtype=np.float)
    feat_dim = feat_pool.shape[1]
    new_trajs = []
    for orig in pool:
        for i in range(samples_per_orig):
            def mean_l1_dist(node, goal):
                node_feats = np.array(node.features, dtype=np.float)
                total = np.linalg.norm(feat_pool - node_feats, 1, axis=1)
                # Can be at most 1 away on each feature dimension, so max norm per point scales in
                # feat dim.
                return feat_dim - total.mean()

            def min_l1_dist(node, goal):
                node_feats = np.array(node.features, dtype=np.float)
                total = np.linalg.norm(feat_pool - node_feats, 1, axis=1)
                # Can be at most 1 away on each feature dimension, so max norm per point scales in
                # feat dim.
                return feat_dim - total.min()

            modificiations = hill_climb(TrajectoryNode(orig),
                                        TrajectoryNode(ANY_PLAN, tuple([1.0] * len(feat_pool[0]))), grid,
                                        min_l1_dist, max_tries=500, tolerance=5.0)
            new = modificiations[-1].trajectory
            print("min l1 v pool: {}".format(abs(min_l1_dist(modificiations[-1], None) - feat_dim)))
            new_trajs.append(new)
            feat_pool = np.vstack([feat_pool, np.array(TrajectoryNode.featurizer(new))])

    return new_trajs, np.array(list(map(TrajectoryNode.featurizer, new_trajs)))


def sample_neighbors(pool, samples_per_orig, grid):
    feat_pool = list(map(TrajectoryNode.featurizer, pool))
    feat_pool = np.array(feat_pool, dtype=np.float)

    new_trajs = []
    for orig in pool:
        for i in range(samples_per_orig):
            traj_feats = TrajectoryNode.featurizer(orig)

            def mean_l1_dist(node, goal):
                node_feats = np.array(node.features, dtype=np.float)
                total = np.linalg.norm(traj_feats - node_feats, 1)
                # We're targeting "nearby" trajectories
                return abs(.3 - total.mean())

            modificiations = hill_climb(TrajectoryNode(orig),
                                        TrajectoryNode(ANY_PLAN, tuple([1.0] * len(feat_pool[0]))), grid,
                                        mean_l1_dist, max_tries=500, tolerance=0.01)
            new = modificiations[-1].trajectory
            print("L1 vs current pool: {}".format(mean_l1_dist(modificiations[-1], None) + .2))
            new_trajs.append(new)
            feat_pool = np.vstack([feat_pool, np.array(TrajectoryNode.featurizer(new), dtype=np.float)])

    return new_trajs, np.array(list(map(TrajectoryNode.featurizer, new_trajs)))


def sample_random_goal_neighbor(pool, samples_per_orig, grid):
    feat_pool = list(map(TrajectoryNode.featurizer, pool))
    feat_pool = np.array(feat_pool, dtype=np.float)
    feat_dim = feat_pool.shape[1]
    new_trajs = []
    for orig in pool:
        orig_feats = TrajectoryNode.featurizer(orig)
        for i in range(samples_per_orig):
            sampled = False
            while not sampled:
                lamb = 1.0
                goal_valid = False
                while not goal_valid:

                    goal = list(orig_feats)
                    random_dim = random.randint(0, len(goal) - 1)
                    sign = -1 if random.random() > 0.5 else 1
                    prev_val = goal[random_dim]
                    mod = sign * random.choice([0.1, 0.2, 0.3])
                    goal[random_dim] = min(1, max(goal[random_dim] + mod, 0))

                    if goal[random_dim] == prev_val:
                        continue
                    goal = TrajectoryNode(ANY_PLAN, tuple(goal))
                    print("Modifying {} by {:.2}".format(random_dim, prev_val - goal.features[random_dim]))
                    goal_valid = True
                succeeded = False
                while not succeeded:
                    print("lamb: {:.3}".format(lamb))
                    directed_l2 = make_directed_l2(np.eye(feat_dim)[random_dim], lamb)

                    modificiations = hill_climb(TrajectoryNode(orig, orig_feats), goal, grid,
                                                directed_l2, max_tries=100, tolerance=0.025)
                    new = modificiations[-1].trajectory
                    new_feats = TrajectoryNode.featurizer(new)
                    result_error = feat_l2(TrajectoryNode(new, new_feats), goal)
                    directed_error = directed_l2(TrajectoryNode(new, new_feats), goal)
                    print("L2 vs goal: {:.2} ({:.2})".format(result_error, directed_error))

                    if directed_error > 0.025:
                        if lamb < 0.1:
                            print("Resampling goal")
                            break
                        lamb *= .50
                        print("trying again")
                        continue

                    print("Got it")
                    succeeded = True
                    sampled = True
                    new_trajs.append(new)
                    feat_pool = np.vstack([feat_pool, np.array(new_feats, dtype=np.float)])

    return new_trajs, np.array(list(map(TrajectoryNode.featurizer, new_trajs)))