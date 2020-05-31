import re
from collections import defaultdict
from copy import copy
import numpy.random as rn
import numpy as np


def knows(knowledge: int, bin_index: int):
    return ((knowledge >> bin_index) & 1) == 1


class HashableState:
    def __init__(self, dict_state):
        self._dict_state = dict_state
        self.hash = self.make_hash(dict_state)

    def freeze(self, o):
        if isinstance(o, dict):
            return frozenset({k: self.freeze(v) for k, v in o.items()}.items())

        if isinstance(o, list):
            return tuple([self.freeze(v) for v in o])

        return o

    def make_hash(self, o):
        """
        makes a hash out of anything that contains only list,dict and hashable types including string and numeric types
        """
        return hash(self.freeze(o))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return isinstance(other, HashableState) and other.hash == self.hash

    def __getitem__(self, item):
        return self._dict_state[item]

    def __str__(self):
        return self._dict_state.__str__()


class BoxInventory:

    def __init__(self, num_bins, requested_bin, discount):
        assert 0 <= requested_bin < num_bins
        self.num_bins = num_bins
        self.requested_bin = requested_bin

        self.actions = ["do_nothing"]
        self.locations = []
        for i in range(self.num_bins):
            name = "navigate" + str(i)
            self.actions.append(name)
            self.locations.append("box" + str(i))
        self.locations.append("human")
        self.actions.append("navigate"+str(i+1))
        self.actions.append("update_human")
        self.locations.append("ABSORBING")
        self.action_name_to_index = {value: key for key, value in enumerate(self.actions)}
        self.n_actions = len(self.actions)
        # State of the robot's knowledge and state of human's knowledge
        # One bit per each bin, so each state is a bitmask

        self.observation_space = {
            "human_knowledge": (self.num_bins ** 2),
            "robot_knowledge": (self.num_bins ** 2),
            "requested_knowledge": (self.num_bins + 1),
            "robot_at": (self.num_bins + 2)
        }
        self._all_states = self._get_states()
        self.n_states = len(self._all_states)
        self.discount = discount

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
        self.living_reward = -1.0
        self.reward_weights = self.calculate_optimal_weights_for_request(self.requested_bin)

    def _transition_model(self, state: dict, action: int) -> dict:
        next_state_dist = defaultdict(lambda: 0.0)
        next_state = copy(state._dict_state)
        if state["robot_at"] == self.locations.index("ABSORBING"):
            # No actions allowed from this state
            next_state_dist[HashableState(next_state)] = 1
            return next_state_dist
        action_name = self.actions[action]
        if "navigate" in action_name:
            location_index = int(re.search("\d$", action_name)[0])
            if location_index < self.num_bins:
                next_state["robot_knowledge"] |= 1 << location_index
            next_state["robot_at"] = location_index
        elif "update_human" == action_name:
            if state["robot_at"] == self.locations.index("human"):
                next_state["human_knowledge"] |= state["robot_knowledge"]
                # Disallow all future actions
                next_state["robot_at"] = self.locations.index("ABSORBING")
            else:
                # Update only works when next to the human
                pass
        elif "do_nothing" == action_name:
            pass
        else:
            assert False
        assert self._is_state_valid(next_state)
        next_state_dist[HashableState(next_state)] = 1.0
        return next_state_dist

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        state = self._all_states[i]
        next_state = self._all_states[k]
        return self._transition_model(state, j)[next_state]

    def _get_states(self):
        states = []
        for h in range(self.observation_space["human_knowledge"]):
            for r in range(self.observation_space["robot_knowledge"]):
                for a in range(self.observation_space["robot_at"]):
                    for k in range(self.observation_space["requested_knowledge"]):
                        if k != self.requested_bin:
                            continue
                        raw_state = {"human_knowledge": h, "robot_knowledge": r, "robot_at": a,
                                     "requested_knowledge": k}
                        if self._is_state_valid(raw_state):
                            states.append(HashableState(raw_state))
        return states

    def state_to_int(self, state):
        return self._all_states.index(state)

    def _get_actions(self):
        return list(range(self.n_actions))

    def _is_state_valid(self, s):
        # Human can't know anything the robot doesn't know
        r_k = s["robot_knowledge"]
        h_k = s["human_knowledge"]
        if not 0 <= r_k < self.observation_space["robot_knowledge"]:
            return False
        if not 0 <= h_k < self.observation_space["human_knowledge"]:
            return False
        for i in range(self.num_bins):
            human_knows_more = ((h_k >> i) & 1) > ((r_k >> i) & 1)
            if human_knows_more:
                return False
            # If the robot's at a spot, it knows what's in the box
            if s["robot_at"] == i and not knows(r_k, i):
                return False
        return True

    def _unpack_state_bitmasks(self, state):
        rk = state["robot_knowledge"]
        unpacked_robot = []
        for i in range(self.num_bins):
            known = (rk >> i) & 1
            if known:
                unpacked_robot.append(i)

        hk = state["human_knowledge"]
        unpacked_human = []
        for i in range(self.num_bins):
            known = (hk >> i) & 1
            if known:
                unpacked_human.append(i)
        unpacked = {"robot_at": self.locations[state["robot_at"]],
                    "robot_knowledge": unpacked_robot,
                    "human_knowledge": unpacked_human,
                    "requested_knowledge": state["requested_knowledge"]}
        return unpacked

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "ident":
            f = np.zeros(self.n_states)
            f[i] = 1
            return f
        else:
            assert False

    def num_state_features(self, feature_map="ident"):
        # TODO: Implement for things besides ident
        if feature_map == "ident":
            return self.n_states
        else:
            assert False

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def reward(self, state: int):
        features = self.feature_vector(state)
        return features.dot(self.reward_weights)

    def _distance_between(self, loc_a, loc_b):
        return 1.0

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                              random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                assert False
            else:
                state = HashableState({"human_knowledge": 0,
            "robot_knowledge": 0,
            "requested_knowledge": self.requested_bin,
            "robot_at": self.locations.index("human")})

            trajectory = []
            for _ in range(trajectory_length):
                # print(self._unpack_state_bitmasks(state._dict_state))
                # Follow the given policy.
                action_num = policy[self.state_to_int(state)]

                next_state_dist = self._transition_model(state, action_num)
                # Deterministic for now
                next_state = [key for key in next_state_dist.keys()][0]
                reward = self.reward(self.state_to_int(next_state))

                state_int = self.state_to_int(state)
                action_int = action_num
                next_state_int = self.state_to_int(next_state)
                trajectory.append((state_int, action_int, reward))
                # print(self.actions[action_num], reward)
                state = next_state

            trajectories.append(trajectory)
        return trajectories

    def optimal_policy_deterministic(self, state):
        state_data = self._all_states[state]
        target_known = knows(state["robot_knowledge"], state["requested_knowledge"])
        # If we know the target information, report
        if target_known:
            return self.action_name_to_index["update_human"]
        else:
            # If we don't find it out
            return self.action_name_to_index["visit"+str(state_data["requested_knowledge"])]

    def calculate_optimal_weights_for_request(self, requested_bin, feature_map="ident"):
        if feature_map == "ident":
            # Each feature describes exactly one state, so the weight is just the reward
            # to give for being in that state
            f = np.zeros(self.n_states)
            for i, state in enumerate(self._all_states):
                if state["robot_at"] == self.locations.index("ABSORBING"):
                    f[i] = 0
                # FIXME(nickswalker): Give reward for being in the absorbing state after delivering
                # correct information. This breaks inf horizon setting
                if knows(state["human_knowledge"], requested_bin):
                    f[i] = 10
                else:
                    f[i] = self.living_reward
            return f
        else:
            assert False


    def print_states_and_transitions(self):
        for state in self._all_states:
            print(state)
            for action in range(self.n_actions):
                next = [key for key in self._transition_model(state, action).keys()][0]
                reward = self.reward(self._all_states.index(next))
                print("\t" + self.actions[action] + ": " + str(next) + " (" + str(reward) + ")")

    def print_trajectory(self, traj):
        states, actions, rewards = traj[:,0].astype(np.int), traj[:,1].astype(np.int), traj[:,2]
        for state, action, reward in zip(states, actions, rewards):
            state = self._all_states[state]
            print(self._unpack_state_bitmasks(state._dict_state))
            print(self.actions[action], reward)
        print("  ")
