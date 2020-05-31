import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import irl.linear_irl as linear_irl
import irl.mdp.gridworld as gridworld
from irl import maxent, value_iteration
from irl.mdp.box_checking import BoxInventory


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


class WeightedStochasticPolicy:
    def __init__(self, weights):
        self.weights = weights
        self.actions = np.arange(len(self.weights[0]))

    def __call__(self, state, **kwargs):
        return np.random.choice(self.actions, p=self.weights[state])


def main(grid_size, discount):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """
    # Make experiments deterministic
    np.random.seed(0)
    gw = BoxInventory(2, 1, discount)
    #gw.print_states_and_transitions()

    # I want a trajectory that maximizes r, but also communicates an attribute to
    # the observer.
    observer = ObserverModel()

    # To do this, uniformly sample weights...
    phi_dim = gw.num_state_features()

    # And solve the MDP under those rewards
    trajs = np.empty([100, 5, 3], dtype=np.float)
    for i in range(100):
        weights = np.random.uniform(np.full([phi_dim], -1), np.full([phi_dim], 1))
        gw.reward_weights = weights
        ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
        opt_action_weights = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.transition_probability, ground_r, gw.discount, stochastic=False)
        #opt_policy = WeightedStochasticPolicy(opt_action_weights)
        trajs[i] = gw.generate_trajectories(1, 5, opt_action_weights)[0]
        gw.print_trajectory(trajs[i])
        print("-")

    # Let's see the attributions for these trajectories
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
        print("ret={},entropy={}:".format(ret, entropy))
        for _, action, _ in traj:
            print(gw.actions[action])


if __name__ == '__main__':
    main(5, 0.2)
