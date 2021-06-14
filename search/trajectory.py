import copy
import queue
import random

import numpy as np

from search.coverage import trajectory_cost
from search.navigation import PointNode
from search.metric import manhattan
from search.routine import astar
from search.util import in_bounds, traversible


class TrajectoryNode:
    featurizer = None

    def __init__(self, value, features, goal_region=None, cost=0):
        self.trajectory = value
        self.features = features
        if features is None:
            self.features = TrajectoryNode.featurizer(value)
        self.parent = None
        self.H = 0
        self.G = cost
        self.goal_region = goal_region

    def move_cost(self, other):
        # Designed to be used with an external cost
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

    def neighbors(self, grid, goal):
        plan = self.trajectory

        # Modify each action in the plan. Repair the plan with shortest path to reconnect.
        neighbors = []
        change_actions(plan, grid, self.goal_region, neighbors)
        shortcuts(plan, grid, self.goal_region, neighbors)
        templates(plan, grid, self.goal_region, neighbors)
        seek_collision(plan, grid, self.goal_region, neighbors)
        overcover(plan, grid, self.goal_region, neighbors)

        return neighbors


def change_actions(plan, grid, goal_region, neighbors):
    width, height = len(grid[0]), len(grid)
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
            # Cut to before the original next point, and then a little further up to cut off the original next point
            prefix, suffix = new_trajectory[:i + 1], new_trajectory[i + 2:]
            connection = astar(PointNode(next_point), PointNode(suffix[0]), grid, manhattan)
            raw_connection = list(map(lambda x: x.point, connection))
            joined = prefix + raw_connection[:-1] + suffix
            # Control  exploration: no trajs longer than 250
            if len(joined) > 250:
                continue
            new_node = TrajectoryNode(joined, None, goal_region=goal_region,
                                      cost=trajectory_cost(goal_region, grid, joined))
            neighbors.append(new_node)


def shortcuts(plan, grid, goal_region, neighbors, length_cap=250):
    # Cuts require 4+ actions to start with
    if len(plan) < 4:
        return
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
        # Control  exploration: no trajs longer than 250
        if len(joined) > length_cap:
            continue
        new_node = TrajectoryNode(joined, None, goal_region=goal_region,
                                  cost=trajectory_cost(goal_region, grid, joined))
        neighbors.append(new_node)


def templates(plan, grid, goal_region, neighbors):
    for i, (x, y) in enumerate(plan):
        if i == 0:
            continue
        # TODO: CHECK BOUND
        if i >= len(plan) - 3:
            break
        orig_prev_point = plan[i - 1]
        delta = (x - orig_prev_point[0], y - orig_prev_point[1])
        straight = (x + delta[0], y + delta[1])
        straight_2 = (straight[0] + delta[0], straight[1] + delta[1])

        # We're already going straight
        if straight_2 == plan[i + 2]:
            continue
        if in_bounds(grid, straight) and in_bounds(grid, straight_2) and traversible(grid, straight) and traversible(
                 grid, straight_2):
            new_trajectory = copy.deepcopy(plan)
            # Cut to before the original next point, and then a little further up to cut off the original next point
            prefix, suffix = new_trajectory[:i + 1] + [straight], new_trajectory[i + 3:]
            connection = astar(PointNode(straight_2), PointNode(suffix[0]), grid, manhattan)
            raw_connection = list(map(lambda x: x.point, connection))
            joined = prefix + raw_connection[:-1] + suffix

            new_node = TrajectoryNode(joined, None, goal_region=goal_region,
                                      cost=trajectory_cost(goal_region, grid, joined))
            neighbors.append(new_node)
        else:
            continue


def seek_collision(plan, grid, goal_region, neighbors):
    for i, (x, y) in enumerate(plan):
        # TODO: Check Bound
        if i >= len(plan) - 2:
            break
        orig_next_point = plan[i + 1]
        neighbor_points = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y), (x, y)]
        # Don't go straight to the next point; that was the original plan...
        neighbor_points.remove(orig_next_point)
        collision_points = [link for link in neighbor_points if grid[link[1]][link[0]] == 'O']

        for next_point in collision_points:
            new_trajectory = copy.deepcopy(plan)
            # Cut to before the original next point, and then a little further up to cut off the original next point
            prefix, suffix = new_trajectory[:i + 1], new_trajectory[i + 2:]
            connection = astar(PointNode(next_point), PointNode(suffix[0]), grid, manhattan)
            raw_connection = list(map(lambda x: x.point, connection))
            joined = prefix + raw_connection[:-1] + suffix
            new_node = TrajectoryNode(joined, None, goal_region=goal_region,
                                      cost=trajectory_cost(goal_region, grid, joined))
            neighbors.append(new_node)


def overcover(plan, grid, goal_region, neighbors):
    for i, (x, y) in enumerate(plan):
        if i < 4:
            continue

        if i >= len(plan) - 3:
            break

        lower_lim = min(3, i)
        upper_lim = min(6, i)
        # As much as 6 steps
        length = random.randrange(lower_lim, upper_lim)

        replay_segment = plan[i - length: i]
        rev_replay_segment = list(reversed(replay_segment))
        # We're already about to retrace our steps
        # TODO: Maybe we want this though?
        if replay_segment == plan[i: i + length]:
            continue

        new_trajectory = copy.deepcopy(plan)
        # Cut to before the original next point, and then a little further up to cut off the original next point
        prefix, suffix = new_trajectory[:i + 1] + rev_replay_segment + replay_segment[1:], new_trajectory[i:]
        joined = prefix + suffix

        new_node = TrajectoryNode(joined, None, goal_region=goal_region,
                                  cost=trajectory_cost(goal_region, grid, joined))
        neighbors.append(new_node)


ANY_PLAN = "any_plan"
