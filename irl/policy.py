import random

import numpy as np


class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, *args, **kwargs):
        return random.randint(0, self.action_space - 1)


class DeterministicPolicy:
    def __init__(self, state_to_action):
        self.state_to_action = state_to_action

    def __call__(self, state, **kwargs):
        return self.state_to_action[state]


class RandDeterministicPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        self._policy = {}

    def __call__(self, state, **kwargs):
        policy_action = self._policy.get(state)
        if policy_action:
            return policy_action
        else:
            self._policy[state] = random.randint(0, self.action_space - 1)
            return self._policy[state]


class WeightedStochasticPolicy:
    def __init__(self, weights):
        self.weights = weights
        self.actions = np.arange(len(self.weights[0]))

    def __call__(self, state, **kwargs):
        return np.random.choice(self.actions, p=self.weights[state])
