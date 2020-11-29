import copy
import random

import numpy as np

from search.navigation import PointNode
from search.metric import manhattan
from search.routine import astar


class TrajectoryNode:

    featurizer = None
    def __init__(self, value, features):
        self.trajectory = value
        self.features = features
        if features is None:
            self.features = TrajectoryNode.featurizer(value)
        self.parent = None
        self.H = 0
        self.G = 0

    def move_cost(self, other):
        return 0

    def __key(self):
        return (tuple(map(tuple, self.trajectory)))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, TrajectoryNode):
            # We have goals that are just feature values
            if self.trajectory == ANY_PLAN:
                node_feats = np.array(other.features, dtype=np.float)
                goal_feats = np.array(self.features, dtype=np.float)
                inds = np.where(np.isnan(goal_feats))
                goal_feats[inds] = np.take(node_feats, inds)
                return (node_feats == goal_feats).all()
            elif other.trajectory == ANY_PLAN:
                node_feats = np.array(other.features, dtype=np.float)
                goal_feats = np.array(self.features, dtype=np.float)
                inds = np.where(np.isnan(node_feats))
                node_feats[inds] = np.take(goal_feats, inds)
                return (node_feats == goal_feats).all()
            else:
                return self.__key() == other.__key()
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, TrajectoryNode):
            return self.__key() < other.__key()
        return NotImplemented

    def neighbors(self, grid):
        plan = self.trajectory
        width, height = len(grid[0]), len(grid)
        # Modify each action in the plan. Repair the plan with shortest path to reconnect.
        neighbors = []
        for i, (x, y) in enumerate(plan):
            if i == len(plan) - 2:
                break
            orig_next_point = plan[i + 1]
            neighbor_points = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y), (x, y)]
            # Don't go straight to the next point; that was the original plan...
            neighbor_points.remove(orig_next_point)
            # Don't go off the grid
            neighbor_points = filter(lambda p: 0 <= p[0] < width and 0 <= p[1] < height, neighbor_points)
            # Eliminate untraversable
            accessible_points = [link for link in neighbor_points if grid[link[1]][link[0]] != 'X']

            for next_point in accessible_points:
                new_trajectory = copy.deepcopy(plan)
                prefix, suffix = new_trajectory[:i + 1], new_trajectory[i + 2:]
                connection = astar(PointNode(next_point), PointNode(suffix[0]), grid, manhattan)
                raw_connection = list(map(lambda x: x.point, connection))
                joined = prefix + raw_connection[:-1] + suffix
                if len(joined) > 250:
                    continue
                new_node = TrajectoryNode(joined, None)
                neighbors.append(new_node)

        # Cuts require 4+ actions to start with
        if len(plan) < 4:
            return neighbors
        to_sample = len(neighbors) // 2
        for _ in range(to_sample):
            cut_start = random.randrange(1, len(plan) - 2)
            cut_end = random.randrange(1, len(plan) - 2)
            if cut_start >= cut_end or cut_end - cut_start < 3:
                continue

            new_trajectory = copy.deepcopy(plan)
            prefix, suffix = new_trajectory[:cut_start], new_trajectory[cut_end:]
            if prefix[-1] == suffix[0]:
                joined = prefix + suffix[1:]
            else:
                connection = astar(PointNode(prefix[-1]), PointNode(suffix[0]), grid, manhattan)
                raw_connection = list(map(lambda x: x.point, connection))
                joined = prefix + raw_connection[1:-1] + suffix
            # Control  exploration
            if len(joined) > 250:
                continue
            new_node = TrajectoryNode(joined, None)
            neighbors.append(new_node)
        return neighbors


ANY_PLAN = "any_plan"