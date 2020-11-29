import copy
from collections import Counter

import numpy as np
from scipy import signal

GOAL_COV = 0
OVERLAP = 1
LENGTH = 2
STRAIGHTNESS = 3
HOOKINESS = 4
CARPET_TIME = 5
COLLISION_TIME = 6
REDUNDANT_COVERAGE = 7
TOTAL_COVERAGE = 8
IDLE_TIME = 9
START_STOPINESS = 10


def trajectory_features(goal, grid, plan):
    # How many unique states do we visit as a percentage of the plan length (0,1]
    self_overlap = 1. - len(set(plan)) / len(plan)
    # [0,1]
    num_missed = len(set(goal).difference(set(plan)))
    goal_cov = 1.0 - (num_missed / len(goal))
    states_in_coverage_zone = [p for p in plan if p in goal]
    covered = Counter(states_in_coverage_zone)
    redundant_coverage = len([v for v in covered.values() if v > 1]) / len(goal)
    # It probably takes ~2x size of goal space number of steps to cover the goal, so let's set 3x as the max
    normalized_plan_length = min(len(plan) / (3 * len(goal)), 1.0)

    points = np.array(plan, dtype=np.int)
    diffs_1 = np.diff(points, axis=0)
    diffs_2 = np.diff(points, 2, axis=0)
    carpet_points = []
    breakage = 0
    for i, point in enumerate(plan):
        contents = grid[point[1]][point[0]]
        if contents == "C":
            carpet_points.append(point)
        elif contents == "O":
            breakage += 1
    carpet_time = min(1.0, len(carpet_points) / len(goal))
    breakage = min(1.0, breakage / len(goal))
    idle_time = (diffs_1 == [0, 0]).all(-1).sum() / len(diffs_1)

    hook_y = signal.correlate2d(diffs_1, np.array([[0, 1], [1, 0], [0, -1]]))
    hook_y_count = (abs(hook_y) == [0, 3, 0]).all(-1).sum()
    hook_x = signal.correlate2d(diffs_1, np.array([[1, 0], [0, 1], [-1, 0]]))
    hook_x_count = (abs(hook_x) == [0, 3, 0]).all(-1).sum()
    start_stop = signal.correlate(abs(diffs_1), np.array([[10, 10], [-10, -10]]))
    start_stop_count = (start_stop[:, 1] == 10).sum()
    start_stopiness = start_stop_count * 2 / len(diffs_1)
    # Each hook template match indicates 3 steps spent "in a turn"
    hookiness = (hook_x_count + hook_y_count) * 3 / len(diffs_2)
    # Second diff of 0 means straight-line motion. Count number of rows with this case and sum
    straight_time = (diffs_2 == [0, 0]).all(-1).sum()
    straightness = straight_time / len(diffs_2)

    covered_x, covered_y = set([p[0] for p in plan]), set([p[1] for p in plan])
    x_coverage = len(covered_x) / len(grid[0])
    y_coverage = len(covered_y) / len(grid)
    total_coverage = len(set(plan)) / (len(grid) * len(grid[0]))

    return (goal_cov,
            self_overlap,
            normalized_plan_length,
            straightness,
            hookiness,
            carpet_time,
            breakage,
            redundant_coverage,
            total_coverage,
            idle_time,
            start_stopiness)


class CoverageNode:
    def __init__(self, value, point, type=None):
        self.to_cover = sorted(value)
        self.point = point
        self.type = type
        self.parent = None
        self.H = 0
        self.G = 0

    def move_cost(self, other):
        # Don't penalize for moves that cover more target cells.
        if len(other.to_cover) < len(self.to_cover):
            return 0
        elif other.type == "O":
            return 5
        else:
            return 1

    def __key(self):
        return (tuple(map(tuple, self.to_cover)), self.point)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, CoverageNode):
            return self.__key() == other.__key()
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, CoverageNode):
            return self.__key() < other.__key()
        return NotImplemented

    def neighbors(self, grid):
        x, y = self.point
        width, height = len(grid[0]), len(grid)
        neighbor_points = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)]
        neighbor_points = filter(lambda p: 0 <= p[0] < width and 0 <= p[1] < height, neighbor_points)
        accessible_points = [link for link in neighbor_points if grid[link[1]][link[0]] != 'X']
        neighbors = []
        for point in accessible_points:
            to_cover = copy.deepcopy(self.to_cover)
            if point in to_cover:
                to_cover.remove(point)
            new_node = CoverageNode(to_cover, point, grid[point[1]][point[0]])
            neighbors.append(new_node)
        return neighbors