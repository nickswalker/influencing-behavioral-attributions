import copy
import itertools
import operator
import os
from collections import Counter
from itertools import product
from queue import PriorityQueue

import pytmx
from functools import partial
import numpy as np
import random
from scipy import signal, spatial
from tqdm import tqdm

random.seed(0)

BREAKABLE = "O"

import shelve


def shelve_it(file_name):
    d = shelve.open(file_name)

    def decorator(func):
        def new_func(*args):
            key = repr(tuple(args))
            if key not in d:
                d[key] = func(*args)
            else:
                print("cache hit")
            return d[key]

        return new_func

    return decorator


class TrajectoryNode:
    featurizer = None

    def __init__(self, value, features=None):
        self.trajectory = value
        self.features = featurizer(value) if value != ANY_PLAN else features
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
                connection = aStar(PointNode(next_point), PointNode(suffix[0]), grid, manhattan)
                raw_connection = list(map(lambda x: x.point, connection))
                joined = prefix + raw_connection[:-1] + suffix
                if len(joined) > 250:
                    continue
                new_node = TrajectoryNode(joined)
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
                connection = aStar(PointNode(prefix[-1]), PointNode(suffix[0]), grid, manhattan)
                raw_connection = list(map(lambda x: x.point, connection))
                joined = prefix + raw_connection[1:-1] + suffix
            # Control  exploration
            if len(joined) > 250:
                continue
            new_node = TrajectoryNode(joined)
            neighbors.append(new_node)
        return neighbors


class PointNode:
    def __init__(self, point, type=None):
        self.point = point
        self.type = type
        self.parent = None
        self.H = 0
        self.G = 0

    def move_cost(self, other):
        if other.type == BREAKABLE:
            return 5
        return 1

    def __key(self):
        return self.point

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, PointNode):
            return self.__key() == other.__key()
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, PointNode):
            return self.__key() < other.__key()
        return NotImplemented

    def neighbors(self, grid):
        x, y = self.point
        width, height = len(grid[0]), len(grid)
        # Yes, we allow the robot to stay in the current location
        neighbor_points = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y), (x, y)]
        neighbor_points = filter(lambda p: 0 <= p[0] < width and 0 <= p[1] < height, neighbor_points)
        accessible_points = [link for link in neighbor_points if grid[link[1]][link[0]] != 'X']
        neighbors = []
        for point in accessible_points:
            new_node = PointNode(point, grid[point[1]][point[0]])
            neighbors.append(new_node)
        return neighbors


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


def aStar(start, goal, grid, heuristic=manhattan):
    # The open and closed sets
    frontier = PriorityQueue()
    openset = set()
    closedset = set()

    # Current point is the starting point
    current = start

    # Add the starting point to the open set
    frontier.put((0, current))
    openset.add(current)
    open_counter = 0

    # While the open set is not empty
    while frontier:
        # Find the item in the open set with the lowest G + H score
        value, current = frontier.get()
        openset.remove(current)

        # If it is the item we want, retrace the path and return it
        if current == goal:
            path = []
            while current.parent:
                path.append(current)
                current = current.parent
            path.append(current)
            # print("Opened {} nodes".format(open_counter))
            return path[::-1]

        # Add it to the closed set
        closedset.add(current)

        # Loop through the node's children/siblings
        for node in current.neighbors(grid):
            # If it is already in the closed set, skip it
            if node in closedset:
                continue

            # Otherwise if it is already in the open set
            if node in openset:
                # Check if we beat the G score
                new_g = current.G + current.move_cost(node)
                if node.G > new_g:
                    # If so, update the node to have a new parent
                    node.G = new_g
                    node.parent = current
            else:
                # If it isn't in the open set, calculate the G and H score for the node
                node.G = current.G + current.move_cost(node)
                node.H = heuristic(node, goal)
                # print(node.G, node.H)

                # Set the parent to our current item
                node.parent = current
                open_counter += 1
                # Add it to the set
                frontier.put((node.G + node.H, node))
                openset.add(node)

    # Throw an exception if there is no path
    raise ValueError('No Path Found')


def hill_climb(start, goal, grid, heuristic=manhattan, tolerance=8, max_tries=1000):
    # The open and closed sets
    frontier = PriorityQueue()
    openset = set()
    closedset = set()

    # Current point is the starting point
    current = start
    current.H = heuristic(current, goal)
    # Add the starting point to the open set
    frontier.put((0, current))
    openset.add(current)
    open_counter = 0

    with tqdm(total=max_tries) as pbar:
        # While the open set is not empty
        while not frontier.empty():
            if len(closedset) >= max_tries:
                current = min(set([*closedset, *openset]), key=lambda o: o.H)
                break
            pbar.update(1)
            # Find the item in the open set with the lowest G + H score
            value, current = frontier.get()

            openset.remove(current)

            # If it is the item we want, retrace the path and return it
            if current == goal or heuristic(current, goal) < tolerance:
                break

            # Add it to the closed set
            closedset.add(current)

            # Loop through the node's children/siblings
            neighbors = current.neighbors(grid)
            # About the practical limit for branching and reasonable search times
            if len(neighbors) > 200:
                neighbors = random.sample(neighbors, 200)
            for node in neighbors:
                # If it is already in the closed set, skip it
                if node in closedset:
                    continue

                # Otherwise if it is already in the open set
                if node in openset:
                    # Check if we beat the G score
                    new_g = current.G + current.move_cost(node)
                    if node.G > new_g:
                        # If so, update the node to have a new parent
                        node.G = new_g
                        node.parent = current
                else:
                    # If it isn't in the open set, calculate the G and H score for the node
                    node.G = current.G + current.move_cost(node)
                    node.H = heuristic(node, goal)
                    # print(node.G, node.H)

                    # Set the parent to our current item
                    node.parent = current
                    open_counter += 1
                    # Add it to the set
                    frontier.put((node.G + node.H, node))
                    openset.add(node)

    path = []
    while current.parent:
        path.append(current)
        current = current.parent
    path.append(current)
    # print("Opened {} nodes".format(open_counter))
    return path[::-1]


@shelve_it("trajectoriesa.shelf")
def get_coverage_plan(start, to_cover, grid_orig):
    return aStar(CoverageNode(to_cover, start), CoverageNode([], start), grid_orig, coverage_manhattan)


def load_map():
    map = pytmx.TiledMap(os.path.abspath("../interface/assets/map32.tmx"))
    width = map.width
    height = map.height
    grid = [[0 for x in range(width)] for y in range(height)]
    collision_layer = map.layers.index(map.get_layer_by_name("walls"))
    deco_layer = map.layers.index(map.get_layer_by_name("deco0"))
    room_layer = map.layers.index(map.get_layer_by_name("bedroom"))
    bedroom = []
    for x, y in product(range(width), range(height)):
        collision = map.get_tile_properties(x, y, collision_layer)
        deco = map.get_tile_properties(x, y, deco_layer)
        room = map.get_tile_properties(x, y, room_layer)
        if room:
            bedroom.append((x, y))
        if collision:
            grid[y][x] = "X"
        elif deco and "type" in deco and "pot" in deco["type"]:
            grid[y][x] = "O"
        elif deco and "type" in deco and "carpet" in deco["type"]:
            grid[y][x] = "C"
        else:
            grid[y][x] = ""
    return grid, bedroom


def pretty_plan(plan):
    if len(plan) == 0:
        return "[]"
    if hasattr(plan[0], "point"):
        plan = map(operator.attrgetter("point"), plan)
        plan = map(str, plan)
    else:
        plan = map(str, plan)
    out = "["
    for node in plan:
        out += node + ", "
    out = out[:-2]
    out += "]"
    return out


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
        for i in range(samples_per_orig):
            sampled = False
            while not sampled:
                lamb = 1.0
                goal_valid = False
                while not goal_valid:

                    goal = list(TrajectoryNode.featurizer(orig))
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

                    modificiations = hill_climb(TrajectoryNode(orig), goal, grid,
                                                directed_l2, max_tries=100, tolerance=0.025)
                    new = modificiations[-1].trajectory
                    result_error = feat_l2(TrajectoryNode(new), goal)
                    directed_error = directed_l2(TrajectoryNode(new), goal)
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
                    feat_pool = np.vstack([feat_pool, np.array(TrajectoryNode.featurizer(new), dtype=np.float)])

    return new_trajs, np.array(list(map(TrajectoryNode.featurizer, new_trajs)))


ANY_PLAN = "any_plan"

if __name__ == '__main__':
    grid, bedroom = load_map()

    plan = get_coverage_plan((10, 9), bedroom, grid)
    print(pretty_plan(plan))

    plan_states = map

    raw_plan = list(map(lambda x: x.point, plan))
    featurizer = partial(trajectory_features, bedroom, grid)
    orig_features = featurizer(raw_plan)
    print(orig_features)
    TrajectoryNode.featurizer = featurizer


    @shelve_it("trajectories.shelf")
    def get_plan_for_feature_goal(goal_feats):
        goal_feats = tuple(goal_feats)
        modificiations = hill_climb(TrajectoryNode(raw_plan), TrajectoryNode(ANY_PLAN, goal_feats), grid, feat_l1,
                                    max_tries=500)
        new_plan = modificiations[-1].trajectory
        print("got within {}.".format(feat_l1(TrajectoryNode(raw_plan), TrajectoryNode(new_plan))))
        return new_plan


    goal_feats = list(orig_features)
    goal_feats[GOAL_COV] = 0
    goal_feats[REDUNDANT_COVERAGE] = None
    goal_feats[TOTAL_COVERAGE] = None
    no_coverage = get_plan_for_feature_goal(goal_feats)
    no_coverage = [(10, 9), (10, 8), (9, 8), (8, 8), (7, 8), (6, 8), (5, 8), (4, 8), (3, 8), (3, 7), (4, 7), (5, 7),
                   (6, 7), (7, 7), (8, 7), (9, 7), (9, 6), (9, 7), (10, 7), (11, 7), (11, 8), (12, 8), (12, 9), (13, 9),
                   (14, 9), (14, 10), (14, 11), (13, 11), (12, 11), (11, 11), (10, 11), (10, 10), (10, 9)]

    # Break something
    goal_feats = list(orig_features)
    goal_feats[COLLISION_TIME] = .1
    goal_feats[TOTAL_COVERAGE] = None
    break_something = get_plan_for_feature_goal(goal_feats)
    break_something = [(10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9), (17, 9), (18, 9), (19, 9),
                       (20, 9), (20, 10), (20, 11), (20, 12), (19, 12), (19, 11), (19, 10), (19, 9), (19, 8), (20, 8),
                       (21, 8), (21, 9), (21, 10), (21, 11), (21, 12), (22, 12), (22, 11), (22, 10), (23, 10), (23, 11),
                       (23, 12), (24, 12), (24, 11), (24, 10), (25, 10), (26, 10), (27, 10), (27, 11), (27, 12),
                       (26, 12), (26, 11), (25, 11), (25, 10), (25, 9), (26, 9), (27, 9), (27, 8), (26, 8), (25, 8),
                       (25, 7), (25, 8), (25, 9), (25, 10), (25, 11), (25, 12), (25, 13), (24, 13), (23, 13), (22, 13),
                       (21, 13), (20, 13), (20, 12), (20, 11), (20, 10), (20, 9), (20, 8), (20, 7), (19, 7), (19, 8),
                       (19, 9), (18, 9), (17, 9), (16, 9), (15, 9), (14, 9), (13, 9), (12, 9), (11, 9), (10, 9)]
    # Explore as much as possible
    goal_feats = [None] * len(orig_features)
    goal_feats[TOTAL_COVERAGE] = 1.0
    # explore_lots = get_plan_for_feature_goal(goal_feats)

    # Make it less wasteful (less overlap)
    goal_feats = list(orig_features)
    goal_feats[REDUNDANT_COVERAGE] = 0
    goal_feats[LENGTH] = 0
    goal_feats[OVERLAP] = 0
    goal_feats[STRAIGHTNESS] = None
    goal_feats[HOOKINESS] = None
    efficient = get_plan_for_feature_goal(goal_feats)
    efficient = [(10, 9), (10, 10), (10, 11), (11, 11), (12, 11), (13, 11), (14, 11), (15, 11), (16, 11), (17, 11),
                 (18, 11), (19, 11), (19, 12), (20, 12), (19, 12), (19, 11), (19, 10), (19, 9), (19, 8), (20, 8),
                 (20, 7), (20, 8), (21, 8), (21, 9), (21, 10), (20, 10), (21, 10), (22, 10), (22, 11), (21, 11),
                 (21, 12), (22, 12), (22, 13), (23, 13), (24, 13), (25, 13), (25, 12), (25, 11), (26, 11), (26, 12),
                 (27, 12), (27, 11), (27, 10), (26, 10), (26, 9), (27, 9), (27, 8), (26, 8), (25, 8), (25, 7), (25, 8),
                 (25, 9), (25, 10), (24, 10), (23, 10), (23, 11), (24, 11), (24, 12), (23, 12), (22, 12), (21, 12),
                 (21, 13), (20, 13), (20, 12), (20, 11), (20, 10), (20, 9), (19, 9), (18, 9), (17, 9), (16, 9), (15, 9),
                 (14, 9), (13, 9), (12, 9), (11, 9), (10, 9)]

    # Try for more turniness
    goal_feats = list(orig_features)
    goal_feats[OVERLAP] = None
    goal_feats[LENGTH] = .6
    goal_feats[HOOKINESS] = 1
    goal_feats[STRAIGHTNESS] = None
    turny = get_plan_for_feature_goal(goal_feats)

    goal_feats = list(orig_features)
    goal_feats[OVERLAP] = None
    goal_feats[LENGTH] = .5
    goal_feats[HOOKINESS] = None
    goal_feats[STRAIGHTNESS] = 1.0
    straight = get_plan_for_feature_goal(goal_feats)

    # Cover quarter of the room only
    goal_feats = [None] * len(orig_features)
    goal_feats[GOAL_COV] = .25
    very_lazy = get_plan_for_feature_goal(goal_feats)
    very_lazy = [(10, 9), (11, 9), (12, 9), (13, 9), (13, 10), (13, 11), (13, 12), (14, 12), (15, 12), (16, 12),
                 (17, 12), (18, 12), (19, 12), (20, 12), (20, 13), (21, 13), (22, 13), (21, 13), (20, 13), (20, 12),
                 (20, 11), (20, 10), (20, 9), (20, 8), (20, 7), (20, 8), (19, 8), (19, 9), (18, 9), (17, 9), (16, 9),
                 (15, 9), (14, 9), (13, 9), (12, 9), (11, 9), (10, 9)]
    # Make it cover twice
    goal_feats = list(orig_features)
    goal_feats[REDUNDANT_COVERAGE] = 1.0
    goal_feats[LENGTH] = None
    goal_feats[OVERLAP] = None
    goal_feats[STRAIGHTNESS] = None
    goal_feats[HOOKINESS] = None
    overkill = get_plan_for_feature_goal(goal_feats)

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
    # Make it start stoppy
    goal_feats = list(orig_features)
    goal_feats[REDUNDANT_COVERAGE] = None
    goal_feats[LENGTH] = .6
    goal_feats[OVERLAP] = None
    goal_feats[STRAIGHTNESS] = None
    goal_feats[START_STOPINESS] = 1.0
    start_stoppy = get_plan_for_feature_goal(goal_feats)

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
    nei_trajs, nei_feats = sample_random_goal_neighbor(pool, 2, grid)
    print("Mean manhat diff: {}".format(spatial.distance.pdist(nei_feats, "cityblock").mean()))

    print(list(itertools.chain(pool, nei_trajs)))
    print("-------")

    print(np.vstack([feat_pool, nei_feats]).tolist())
