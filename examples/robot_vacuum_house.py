import copy
import os
from itertools import product
import time
from queue import PriorityQueue

import pytmx


class Node:
    def __init__(self, value, point):
        self.value = sorted(value)
        self.point = point
        self.parent = None
        self.H = 0
        self.G = 0

    def move_cost(self, other):
        # Don't penalize for moves that cover more target cells.
        return 0 if len(other.value) < len(self.value) else 1

    def __key(self):
        return (tuple(map(tuple,self.value)), self.point)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.__key() == other.__key()
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.__key() < other.__key()
        return NotImplemented


def neighbors(node, grid):
    x, y = node.point
    width, height = len(grid[0]), len(grid)
    neighbor_points = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)]
    neighbor_points = filter(lambda p: 0 <= p[0] < width and 0 <= p[1] < height, neighbor_points)
    accessible_points = [link for link in neighbor_points if grid[link[1]][link[0]] != 'X']
    neighbors = []
    for point in accessible_points:
        to_cover = copy.deepcopy(node.value)
        if point in to_cover:
            to_cover.remove(point)
        new_node = Node(to_cover, point)
        neighbors.append(new_node)
    return neighbors

def manhattan(node, goal):
    dists = []
    for to_cover in node.value:
        dists.append(abs(to_cover[0] - node.point[0]) + abs(to_cover[1] - node.point[0]))

    if len(dists) == 0:
        dists.append(abs(node.point[0] - goal.point[0]) + abs(node.point[1] - goal.point[1]))
    else:
        # How long is it going to take to return to the start position
        furthest_point = node.value[dists.index(max(dists))]
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
            print("Opened {} nodes".format(open_counter))
            return path[::-1]

        # Add it to the closed set
        closedset.add(current)

        # Loop through the node's children/siblings
        for node in neighbors(current, grid):
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
                print(node.G, node.H)

                # Set the parent to our current item
                node.parent = current
                open_counter += 1
                # Add it to the set
                frontier.put((node.G + node.H, node))
                openset.add(node)

    # Throw an exception if there is no path
    raise ValueError('No Path Found')


def get_coverage_plan(start, to_cover, grid_orig):
    return aStar(Node(to_cover, start), Node([], start), grid_orig)


def load_map():
    map = pytmx.TiledMap(os.path.abspath("../interface/assets/map32.tmx"))
    width = map.width
    height = map.height
    grid = [[0 for x in range(width)] for y in range(height)]
    collision_layer = map.layers.index(map.get_layer_by_name("collision"))
    room_layer = map.layers.index(map.get_layer_by_name("bedroom"))
    bedroom = []
    for x, y in product(range(width), range(height)):
        contents = map.get_tile_properties(x, y, collision_layer)
        room = map.get_tile_properties(x, y, room_layer)
        if room:
            bedroom.append((x,y))
        grid[y][x] = "X" if contents else " "
    return grid, bedroom


def pretty_plan(plan):
    out = ""
    for node in plan:
        out += str(node.point) + " "
    return out


if __name__ == '__main__':
    grid, bedroom = load_map()

    start = time.time()
    plan = get_coverage_plan((10,9), bedroom, grid)
    end = time.time()
    print(end - start)
    print(pretty_plan(plan))
