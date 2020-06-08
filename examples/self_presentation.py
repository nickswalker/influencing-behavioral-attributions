import numpy as np
import scipy.stats

import irl.mdp.gridworld as gridworld

from irl import maxent, value_iteration
from irl.policy import DeterministicPolicy
from observer.observer import HugsWall


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))


def main(grid_size, discount):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    # Make experiments deterministic
    np.random.seed(0)
    gw = gridworld.Gridworld(5, 0, .1)
    #gw = BoxInventory(2, 1, discount)
    #gw.print_states_and_transitions()

    # I want a trajectory that maximizes r, but also communicates an attribute to
    # the observer.
    observer = HugsWall(5)
    phi_dim = gw.num_state_features(feature_map="ident")
    # To do this, uniformly sample weights and solve the MDP under those rewards
    weights_to_try = [gw.reward_weights]

    for i in range(50):
        weights_to_try.append(np.random.uniform(np.full([phi_dim], -1), np.full([phi_dim], 1)) * 0.2 + gw.reward_weights)
    # Single state activations
    weights_to_try += np.eye(gw.num_state_features(feature_map="ident")).tolist()
    for i in range(300):
        weights_to_try.append(np.random.uniform(np.full([phi_dim], -1), np.full([phi_dim], 1)))

    trajs = np.empty([len(weights_to_try), 20, 3], dtype=np.float)
    for i, weights in enumerate(weights_to_try):
        gw.reward_weights = weights
        ground_r = np.array([gw.reward(s, feature_map="ident") for s in range(gw.n_states)])
        opt_action_weights = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.transition_probability, ground_r, gw.discount, stochastic=False)
        #opt_policy = WeightedStochasticPolicy(opt_action_weights)
        det_policy = DeterministicPolicy(opt_action_weights)
        trajs[i] = gw.generate_trajectories(1, 20, det_policy)[0]
        gw.print_trajectory(trajs[i])
        print("-")

    # Let's see the attributions for these trajectories
    weights = []
    likelihoods = []
    for traj in trajs:
        # Returns can be floats but we just need state/action indices
        traj_int = np.expand_dims(traj, 0).astype(int)
        r = maxent.irl(gw.feature_matrix("gdist"), gw.n_actions, gw.discount, gw.transition_probability,
                           traj_int, 10, 0.1)
        opt_action_weights = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.transition_probability, r,
                                                         gw.discount, stochastic=False)
        det_policy = DeterministicPolicy(opt_action_weights)
        # Informational: Opt-trajectory given the extracted reward function
        opt_traj = gw.generate_trajectories(1, 20, det_policy)[0]
        # Now lets see what our attribution model says
        likelihood = observer.p_behavior_given_theta(r, feature_map="gdist")
        weights.append(r)
        likelihoods.append(likelihood)

    # Sum of rewards (not discounted)
    returns = trajs.sum(axis=1)[:,2]
    entropy_scores = np.empty(len(trajs))
    for i in range(len(entropy_scores)):
        entropy_scores[i] = scipy.stats.entropy(likelihoods[i], [1, 0, 0])

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
