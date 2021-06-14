import operator
import os
import shelve
from itertools import product

import pytmx


def load_map(path):
    map = pytmx.TiledMap(os.path.abspath(path))
    width = map.width
    height = map.height
    grid = [[0 for x in range(width)] for y in range(height)]
    collision_layer = map.layers.index(map.get_layer_by_name("collision"))
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


def in_bounds(grid, point):
    width, height = len(grid[0]), len(grid)
    return point[0] < width and 0 <= point[1] < height


def traversible(grid, point):
    return grid[point[1]][point[0]] != 'X'


BREAKABLE = "O"
