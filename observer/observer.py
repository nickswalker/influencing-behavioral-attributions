import numpy as np
from scipy import stats

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class ObserverModel:
    def __init__(self):
        pass

    def p_behavior_given_theta(self, theta):
        # How likely is a behavioral class for reward params theta?
        raise NotImplementedError()


class HugsWall:
    def __init__(self, grid_size):
        self.attributions = ['hugs'] + ['0']
        self.grid_size = grid_size

    def p_behavior_given_theta(self, theta, feature_map="ident"):
        if feature_map == "ident":
            return self.p_behavior_given_theta_ident(theta)
        if feature_map == "coord":
            return self.p_behavior_given_theta_coord(theta)
        raise NotImplementedError()

    def p_behavior_given_theta_coord(self, theta):
        return [1, 0]

    def p_behavior_given_theta_ident(self, theta):
        overall_mean = np.mean(theta)
        # Meant for proxi or ident features
        wall_vals = []
        inside_vals = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                hugging = False
                # At the top or bottom?
                if i == self.grid_size - 1 or i == 0:
                    hugging = True
                # At the left or right
                if j == self.grid_size - 1 or j == 0:
                    hugging = True
                val = theta[i * self.grid_size + j]
                if hugging:
                    wall_vals.append(val)
                else:
                    inside_vals.append(val)

        wall_mean, inside_mean = np.mean(wall_vals), np.mean(inside_vals)
        confidence_in_wall = sigmoid(wall_mean - inside_mean)
        return [confidence_in_wall, 1 - confidence_in_wall]
