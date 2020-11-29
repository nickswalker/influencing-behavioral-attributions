import random
from queue import PriorityQueue

from tqdm import tqdm

from search.metric import manhattan


def astar(start, goal, grid, heuristic=manhattan):
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