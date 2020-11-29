from search.util import BREAKABLE


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