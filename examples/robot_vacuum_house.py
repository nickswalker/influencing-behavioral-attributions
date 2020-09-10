import copy
import os
import random
from itertools import product
import time
from queue import PriorityQueue

import pytmx
from functools import partial
from operator import sub
import numpy as np

BREAKABLE = "O"


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
            neighbor_points = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)]
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
                new_node = TrajectoryNode(joined)
                neighbors.append(new_node)

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
        neighbor_points = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)]
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


# It's not obvious how to handle trajectory features because some can't be computed until the full trajectory
# is known
class MultiObjectiveCoverageNode:
    def __init__(self, value, point):
        self.to_cover = sorted(value)
        self.point = point
        self.parent = None
        self.H = 0
        self.G = 0

    def move_cost(self, other):
        # Don't penalize for moves that cover more target cells.
        return 0 if len(other.to_cover) < len(self.to_cover) else 1

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
            new_node = CoverageNode(to_cover, point)
            neighbors.append(new_node)
        return neighbors


def manhattan(node, goal):
    return abs(goal.point[0] - node.point[0]) + abs(goal.point[1] - node.point[0])


def feat_diff(node, goal):
    node_feats = np.array(node.features, dtype=np.float)
    goal_feats = np.array(goal.features, dtype=np.float)
    inds = np.where(np.isnan(goal_feats))
    goal_feats[inds] = np.take(node_feats, inds)
    return 100 * abs(node_feats - goal_feats).sum()


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


def hill_climb(start, goal, grid, heuristic=manhattan, tolerance=8, max_tries=100):
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

    # While the open set is not empty
    while frontier:
        if len(closedset) > max_tries:
            current = min(set([*closedset, *openset]), key=lambda o: o.H)
            print("Giving up")
            break
        # Find the item in the open set with the lowest G + H score
        value, current = frontier.get()

        openset.remove(current)

        # If it is the item we want, retrace the path and return it
        if current == goal or heuristic(current, goal) < tolerance:
            break

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

    path = []
    while current.parent:
        path.append(current)
        current = current.parent
    path.append(current)
    print("Opened {} nodes".format(open_counter))
    return path[::-1]


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
    out = ""
    for node in plan:
        out += str(node.point) + " "
    return out


def trajectory_features(goal, grid, plan):
    # How many unique states do we visit as a percentage of the plan length (0,1]
    self_overlap = len(set(plan)) / len(plan)
    # [0,1]
    num_missed = len(set(goal).difference(set(plan)))
    coverage = 1.0 - (num_missed / len(goal))
    # It probably takes ~2x size of goal space number of steps to cover the goal, so let's set 3x as the max
    normalized_plan_length = min(len(plan) / (3 * len(goal)), 1.0)

    carpet_points = []
    deltas = []
    breakage = 0
    for i, point in enumerate(plan):
        if i == 0:
            continue
        prev = plan[i - 1]
        deltas.append((point[0] - prev[0], point[1] - prev[1]))
        contents = grid[point[1]][point[0]]
        if contents == "C":
            carpet_points.append(point)
        elif contents == "O":
            breakage += 1
    carpet_time = len(carpet_points)

    turns = 0
    for i, delta in enumerate(deltas):
        if i == 0:
            continue
        if deltas[i - 1] == deltas[i]:
            turns += 1
    turniness = turns / len(deltas)

    return (self_overlap, coverage, normalized_plan_length, turniness, carpet_time, breakage)


ANY_PLAN = "any_plan"

if __name__ == '__main__':
    grid, bedroom = load_map()

    start = time.time()
    plan = get_coverage_plan((10, 9), bedroom, grid)
    end = time.time()
    print(end - start)
    print(pretty_plan(plan))

    plan_states = map

    raw_plan = list(map(lambda x: x.point, plan))
    featurizer = partial(trajectory_features, bedroom, grid)
    orig_features = featurizer(raw_plan)
    print(featurizer(raw_plan))
    TrajectoryNode.featurizer = featurizer
    goal_feats = list(orig_features)
    # Make it less wasteful (less overlap)
    # goal_feats[0] = 0
    # goal_feats[3] = None
    # goal_feats[4] = None

    # Try for more turniness
    goal_feats[0] = None
    goal_feats[2] = None
    goal_feats[3] = 1

    # Cover quarter of the room only
    # goal_feats[1] = .25

    # goal_feats[1] = .5

    # goal_feats[1] = .75

    # Don't cover carpet.
    # No constraint on turniness or overlap
    # goal_feats[0] = None
    # goal_feats[4] = 0
    # goal_feats[3] = None

    # Break something
    # goal_feats[5] = 3

    goal_feats = tuple(goal_feats)
    modificiations = hill_climb(TrajectoryNode(raw_plan), TrajectoryNode(ANY_PLAN, goal_feats), grid, feat_diff)
    print(modificiations[-1].trajectory)
    print(featurizer(modificiations[-1].trajectory))
