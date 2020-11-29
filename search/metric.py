import numpy as np


def manhattan(node, goal):
    return abs(goal.point[0] - node.point[0]) + abs(goal.point[1] - node.point[0])


def feat_l1(node, goal):
    node_feats = np.array(node.features, dtype=np.float)
    goal_feats = np.array(goal.features, dtype=np.float)
    inds = np.where(np.isnan(goal_feats))
    goal_feats[inds] = np.take(node_feats, inds)
    return np.linalg.norm(node_feats - goal_feats, 1)


def feat_l2(node, goal):
    node_feats = np.array(node.features, dtype=np.float)
    goal_feats = np.array(goal.features, dtype=np.float)
    inds = np.where(np.isnan(goal_feats))
    goal_feats[inds] = np.take(node_feats, inds)
    return np.linalg.norm(node_feats - goal_feats, 2)


def make_directed_l2(direction, lamb):
    def feat_l2_directed(node, goal):
        node_feats = np.array(node.features, dtype=np.float)
        goal_feats = np.array(goal.features, dtype=np.float)
        inv_dir = np.logical_not(direction).astype(int)
        return np.linalg.norm((node_feats - goal_feats) * direction) + lamb * np.linalg.norm(
            (node_feats - goal_feats) * inv_dir)

    return feat_l2_directed


def coverage_manhattan(node, goal):
    dists = []
    for to_cover in node.to_cover:
        dists.append(abs(to_cover[0] - node.point[0]) + abs(to_cover[1] - node.point[0]))

    if len(dists) == 0:
        dists.append(abs(node.point[0] - goal.point[0]) + abs(node.point[1] - goal.point[1]))
    else:
        # How long is it going to take to return to the start position
        furthest_point = node.to_cover[dists.index(max(dists))]
        dists.append(abs(furthest_point[0] - goal.point[0]) + abs(furthest_point[1] - goal.point[1]))
    return min(dists) + len(dists)