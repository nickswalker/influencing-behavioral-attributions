from queue import Queue

import numpy as np
import scipy.stats

import irl.mdp.gridworld as gridworld

from irl import maxent, value_iteration
from irl.policy import DeterministicPolicy
from observer.observer import HugsWall


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))


def sample_reward_weights(n, phi_dim, task_weights):
    weights = []
    for i in range(n // 2):
        weights.append(np.random.uniform(np.full([phi_dim], -1), np.full([phi_dim], 1)) * 0.2 + task_weights)
        # Single state activations
    weights += np.eye(phi_dim).tolist()
    for i in range(n // 2):
        weights.append(np.random.uniform(np.full([phi_dim], -1), np.full([phi_dim], 1)))
    return weights


def sample_policies(optimal_policy):
    pass


def sample_trajectories_near_optimal(gw):
    trajectories = []
    traj_length = 10
    ground_r = np.array([gw.reward(s, feature_map="ident") for s in range(gw.n_states)])
    opt_action_weights = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.transition_probability, ground_r,
                                                     gw.discount, stochastic=False)
    opt_policy = DeterministicPolicy(opt_action_weights)
    opt_traj = gw.generate_trajectories(1, traj_length, opt_policy)[0]
    to_mutate = Queue()
    to_mutate.put((opt_action_weights, opt_traj))
    result = []
    seen = set()
    while not to_mutate.empty():
        weights, traj = to_mutate.get()
        modified_trajs = []
        for i in range(traj_length):
            # change one action in policy
            state_i, action_i = int(traj[i][0]), int(traj[i][1])
            new_weights = weights.copy()
            for new_a in range(gw.n_actions):
                if new_a == action_i:
                    continue
                # Change one action selection
                new_weights[state_i] = new_a
                new_policy = DeterministicPolicy(new_weights)
                new_traj = gw.generate_trajectories(1, traj_length, new_policy)[0]
                # Check that both end in the same place. If not, skip this mutation
                if new_traj[-1][0] != opt_traj[-1][0]:
                    continue
                # No duplicates
                if hash(new_traj.data.tobytes()) in seen:
                    continue
                seen.add(hash(new_traj.data.tobytes()))
                modified_trajs.append(new_traj)
                to_mutate.put((new_weights, new_traj))

        result += modified_trajs

    return result


def trajectories_for_weights(weights_to_try, gw):
    orig_weights = gw.reward_weights
    traj_length = 20
    trajs = np.empty([len(weights_to_try), traj_length, 3], dtype=np.float)

    for i, weights in enumerate(weights_to_try):
        gw.reward_weights = weights
        ground_r = np.array([gw.reward(s, feature_map="ident") for s in range(gw.n_states)])
        opt_action_weights = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.transition_probability, ground_r,
                                                         gw.discount, stochastic=False)
        # opt_policy = WeightedStochasticPolicy(opt_action_weights)
        det_policy = DeterministicPolicy(opt_action_weights)
        trajs[i] = gw.generate_trajectories(1, traj_length, det_policy)[0]
        gw.print_trajectory(trajs[i])
        print("-")
    gw.reward_weights = orig_weights
    return trajs


def main(grid_size, discount):
    # Make experiments deterministic
    np.random.seed(0)
    gw = gridworld.Gridworld(5, 0, .1)
    # gw = BoxInventory(2, 1, discount)
    # gw.print_states_and_transitions()

    # I want a trajectory that maximizes r, but also communicates an attribute to
    # the observer.

    # A model that scores attributions given reward parameters
    observer = HugsWall(5)
    phi_dim = gw.num_state_features(feature_map="ident")
    task_weights = gw.reward_weights
    """
    # To do this, sample state-feature weightings
    weights_to_try = sample_reward_weights(500, phi_dim, task_weights)

    # Unmodified task weights
    weights_to_try.append(task_weights)

    # Get optimal policy trajectories based on these weightings
    trajs = trajectories_for_weights(weights_to_try, gw)
    """

    trajs = sample_trajectories_near_optimal(gw)

    # Let's see the attributions for these trajectories
    weights = []
    likelihoods = []
    opt_trajs = []
    for traj in trajs:
        # Returns can be floats but we just need state/action indices
        traj_int = np.expand_dims(traj, 0).astype(int)
        r = maxent.irl(gw.feature_matrix("ident"), gw.n_actions, gw.discount, gw.transition_probability,
                       traj_int, 100, 0.1)
        opt_action_weights = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.transition_probability, r,
                                                         gw.discount, stochastic=False)
        det_policy = DeterministicPolicy(opt_action_weights)
        # Informational: Opt-trajectory given the extracted reward function
        opt_traj = gw.generate_trajectories(1, 10, det_policy)[0]
        opt_trajs.append(opt_traj)
        # Now lets see what our attribution model says
        likelihood = observer.p_behavior_given_theta(r, feature_map="ident")
        weights.append(r)
        likelihoods.append(likelihood)

    # Sum of rewards (not discounted)
    returns = [traj.sum(axis=0)[2] for traj in trajs]
    entropy_scores = np.empty(len(trajs))
    for i in range(len(entropy_scores)):
        entropy_scores[i] = scipy.stats.entropy(likelihoods[i], [.99, .01])

    # Penalize high entropy
    combined_scores = returns + -entropy_scores
    sorted_low_to_high = np.argsort(combined_scores)
    for i in range(5):
        index = sorted_low_to_high[-1 - i]
        traj, ret, entropy = trajs[index], returns[index], entropy_scores[index]
        print("ret={},entropy={}:".format(ret, entropy))
        gw.print_trajectory(traj)
        gw.print_trajectory(opt_trajs[i])


if __name__ == '__main__':
    main(5, 0.2)
