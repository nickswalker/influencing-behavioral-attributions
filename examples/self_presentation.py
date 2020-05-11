import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import irl.linear_irl as linear_irl
import irl.mdp.gridworld as gridworld
from irl import maxent


class ObserverModel:
    def __init__(self):
        self.attributions = ['a', 'b'] + ['0']

    def p_behavior_given_theta(self, theta):
        # How likely is a behavioral class for reward params theta?
        # This would need to be empirically modeled, but for this toy
        # example, we'll set a threshold on how some of the features
        # are weighted
        if theta[3] > .5:
            return [1, 0, 0]
        elif theta[4] > .5:
            return [0, 1, 0]
        else:
            # Put all mass on the UNK class
            return [0] * (len(self.attributions) - 1) + [1]


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))


class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, *args, **kwargs):
        return random.randint(0, self.action_space - 1)


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


def main(grid_size, discount):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    wind = 0.0
    gw = gridworld.Gridworld(grid_size, wind, discount)

    # I want a trajectory that maximizes r, but also communicates an attribute to
    # the observer.
    observer = ObserverModel()

    # To do this, sample a lot of trajectories
    # rand_policy = RandPolicy(gw.n_actions)
    # Random policy is not a good way to generate trajectories...
    #trajs = gw.generate_trajectories(100, 10, rand_policy)
    # state action reward tuples
    trajs = np.empty([100, 10, 3], dtype=np.int)
    for i in range(100):
        det_rand_policy = RandDeterministicPolicy(gw.n_actions)
        trajs[i] = gw.generate_trajectories(1, 10, det_rand_policy)[0]

    # Throw in some optimal policies too
    opt_trajs = np.empty([5, 10, 3], dtype=np.int)
    for i in range(5):
        # TODO: This is always the same det opt policy
        opt_trajs[i] = gw.generate_trajectories(1, 10, gw.optimal_policy_deterministic)[0]
    trajs = np.vstack([trajs, opt_trajs])
    # Let's see what the reward weights would be for these trajectories
    weights = []
    likelihood = []
    for traj in trajs:
        traj = np.expand_dims(traj, 0)
        r = maxent.irl(gw.feature_matrix(), gw.n_actions, gw.discount, gw.transition_probability,
                           traj, 10, 0.1)
        weights.append(r)
        # Now lets score them by likelihood of being attributed as class 'a'
        likelihood.append(observer.p_behavior_given_theta(r))

    # Sum of rewards (not discounted)
    returns = trajs.sum(axis=1)[:,2]
    entropy_scores = np.empty(len(trajs))
    for i in range(len(entropy_scores)):
        entropy_scores[i] = scipy.stats.entropy(likelihood[i], [1, 0, 0])

    # Penalize high entropy
    combined_scores = returns + -entropy_scores
    sorted_low_to_high = np.argsort(combined_scores)
    for i in range(5):
        index = sorted_low_to_high[-1 - i]
        traj, ret, entropy = trajs[index], returns[index], entropy_scores[index]
        print("ret={},entropy={}: {}", ret, entropy)
        for _, action, _ in traj:
            print(gw.actions[action])


if __name__ == '__main__':
    main(5, 0.2)
