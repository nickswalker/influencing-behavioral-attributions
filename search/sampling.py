import random

import numpy as np
from joblib import delayed, Parallel

from search.trajectory import TrajectoryNode, ANY_PLAN
from search.metric import make_directed_l2, feat_l2
from search.routine import hill_descend


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

            modificiations = hill_descend(TrajectoryNode(orig),
                                          TrajectoryNode(ANY_PLAN, tuple([1.0] * len(feat_pool[0]))), grid,
                                          min_l1_dist, max_tries=500, tolerance=5.0)
            new = modificiations[-1].trajectory
            print("min l1 v pool: {}".format(abs(min_l1_dist(modificiations[-1], None) - feat_dim)))
            new_trajs.append(new)
            feat_pool = np.vstack([feat_pool, np.array(TrajectoryNode.featurizer(new))])

    return new_trajs, np.array(list(map(TrajectoryNode.featurizer, new_trajs)))


def sample_neighbors_for_pool(pool, samples_per, grid, featurizer, target_distance=[0.1,0.3]):
    return Parallel(n_jobs=16,verbose=10)(delayed(sample_neighbors)(traj, samples_per, grid, featurizer, target_distance=random.uniform(*target_distance)) for traj in pool)


def sample_neighbors(orig, samples, grid, featurizer, target_distance=0.3):
    new_trajs = []
    # We might be a new joblib job, so ensure this hacked global state is configured
    TrajectoryNode.featurizer = featurizer
    for i in range(samples):
        traj_feats = featurizer(orig)

        def mean_l1_dist(node, goal):
            node_feats = np.array(node.features, dtype=np.float)
            total = np.linalg.norm(traj_feats - node_feats, 1)
            # We're targeting "nearby" trajectories, so try to drive dist to the target to 0
            return abs(target_distance - total.mean())

        modificiations = hill_descend(TrajectoryNode(orig, None),
                                      TrajectoryNode(ANY_PLAN, (None,)), grid,
                                      mean_l1_dist, max_tries=500, tolerance=0.1, verbose=False)
        new = modificiations[-1].trajectory
        #print("L1 new v orig: {}".format(mean_l1_dist(modificiations[-1], None) + target_distance))
        new_trajs.append(new)
    return new_trajs


def sample_perturb_neighbor_pool(pool, samples_per_orig, grid, featurizer):
    return Parallel(n_jobs=20,verbose=50)(delayed(sample_perturb_neighbor)(traj, samples_per_orig, grid, featurizer) for traj in pool)


def sample_perturb_neighbor(orig, samples, grid, featurizer, verbose=False):
    TrajectoryNode.featurizer = featurizer
    trajs = []
    orig_feats = featurizer(orig)
    feat_dim = len(orig_feats)
    for i in range(samples):
        sampled = False
        tries = 0
        while not sampled and tries < 3:
            lamb = 1.0
            goal_valid = False
            while not goal_valid:

                goal = list(orig_feats)
                random_dim = random.randint(0, len(goal) - 1)
                sign = -1 if random.random() > 0.5 else 1
                prev_val = goal[random_dim]
                mod = sign * random.uniform(0.1, 0.3)
                goal[random_dim] = min(1, max(goal[random_dim] + mod, 0))

                if goal[random_dim] == prev_val:
                    continue
                goal = TrajectoryNode(ANY_PLAN, tuple(goal))
                #print("Modifying {} by {:.2}".format(random_dim, prev_val - goal.features[random_dim]))
                goal_valid = True
            succeeded = False
            while not succeeded:
                #print("lamb: {:.3}".format(lamb))
                directed_l2 = make_directed_l2(np.eye(feat_dim)[random_dim], lamb)

                modificiations = hill_descend(TrajectoryNode(orig, orig_feats), goal, grid,
                                              directed_l2, max_tries=100, tolerance=0.025, verbose=verbose)
                new = modificiations[-1].trajectory
                new_feats = TrajectoryNode.featurizer(new)
                result_error = feat_l2(TrajectoryNode(new, new_feats), goal)
                directed_error = directed_l2(TrajectoryNode(new, new_feats), goal)
                #print("L2 vs goal: {:.2} ({:.2})".format(result_error, directed_error))

                if directed_error > 0.025:
                    if lamb < 0.1:
                        #print("Resampling goal")
                        tries += 1
                        break
                    lamb *= .50
                    #print("trying again")
                    continue

                trajs.append(new)
                #print("Got it")
                succeeded = True
                sampled = True
        if tries == 4:
            print("gave up")


    return trajs
