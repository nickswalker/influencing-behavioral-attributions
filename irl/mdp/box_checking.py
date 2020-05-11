import re
from copy import copy


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


class BoxInventory:
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_bins = 8

        self.action_to_name = {0: "do_nothing"}
        self.location_to_name = {0: "human"}
        for i in range(1, self.num_bins + 1):
            name = "visit" + str(i)
            self.action_to_name[i] = name
            self.location_to_name[i] = "box" + str(i)
        self.action_to_name[i + 1] = "update_human"
        self.action_name_to_index = {value: key for key, value in self.action_to_name.items()}
        self.action_space = Discrete(len(self.action_to_name))
        # State of the robot's knowledge and state of human's knowledge
        # One bit per each bin, so each state is a bitmask
        self.observation_space = Dict({
            "human_knowledge": Discrete(self.num_bins ** 2),
            "robot_knowledge": Discrete(self.num_bins ** 2),
            "requested_knowledge": Discrete(self.num_bins + 1),
            "robot_at": Discrete(self.num_bins + 1)
        })
        self.living_reward = -1.0
        self.state = {}
        self.counter = 0
        self.done = 0
        self.add = [0, 0]
        self.reward = 0
        self.reset()

    def check_episode_done(self, state):
        requested = state["requested_knowledge"]
        return bool(state["human_knowledge"] & (1 << requested))

    def step(self, action):
        if self.done == True:
            print("Game Over")
            return [self.state, self.reward, self.done, self.add]

        next_state_dist = self._transition_model(self.state, action)
        # Deterministic for now
        next_state = [key for key in next_state_dist.keys()][0]
        self.reward = self._calculate_reward(self.state, action, next_state)
        self.state = next_state
        self.render()

        self.done = self.check_episode_done(self.state)
        return [self.state, self.reward, self.done, self.add]

    def _transition_model(self, state, action):
        next_state_dist = {}
        next_state = copy(state._dict_state)
        action_name = self.action_to_name[action]
        if "visit" in action_name:
            target_bin = int(re.search("\d$", action_name)[0])
            next_state["robot_knowledge"] |= 1 << target_bin
            next_state["robot_at"] = target_bin
        elif "update_human" == action_name:
            next_state["human_knowledge"] |= state["robot_knowledge"]
            next_state["robot_at"] = 0
        elif "do_nothing" == action_name:
            pass
        else:
            assert False
        next_state_dist[HashableState(next_state)] = 1.0
        return next_state_dist

    def _get_states(self):
        states = []
        for h in range(self.observation_space["human_knowledge"].n):
            for r in range(self.observation_space["robot_knowledge"].n):
                for a in range(self.observation_space["robot_at"].n):
                    for k in range(self.observation_space["requested_knowledge"].n):
                        raw_state = {"human_knowledge": h, "robot_knowledge": r, "robot_at": a,
                                     "requested_knowledge": k}
                        if self._is_state_valid(raw_state):
                            states.append(HashableState(raw_state))
        return states

    def _get_actions(self):
        return list(range(self.action_space.n))

    def _is_state_valid(self, s):
        # Human can't know anything the robot doesn't know
        r_k = s["robot_knowledge"]
        h_k = s["human_knowledge"]
        for i in range(self.num_bins):
            human_knows_more = ((h_k >> i) & 1) > ((r_k >> i) & 1)
            if human_knows_more:
                return False
        return True

    def _calculate_reward(self, state, action, next_state):
        human_knowledge_gained = 0
        robot_knowledge_gained = 0
        dist_traveled = self._distance_between(state["robot_at"], next_state["robot_at"])
        robot_changed = state["robot_knowledge"] ^ next_state["robot_knowledge"]
        human_changed = state["human_knowledge"] ^ next_state["human_knowledge"]
        request_fufilled = bool(human_changed & (1 << state["requested_knowledge"]))
        for i in range(self.num_bins):
            # Note: This actually just checks what changed, assuming that knowledge never goes away...
            robot_knowledge_gained += (robot_changed >> i) & 1
            human_knowledge_gained += (human_changed >> i) & 1
        reward = self.living_reward + human_knowledge_gained + robot_knowledge_gained - dist_traveled
        if request_fufilled:
            reward += 10
        return reward

    def reset(self):
        self.state["human_knowledge"] = 0
        self.state["robot_knowledge"] = 0
        self.state["requested_knowledge"] = 0
        self.counter = 0
        self.done = 0
        self.add = [0, 0]
        self.reward = 0
        return self.state

    def _distance_between(self, loc_a, loc_b):
        return 1.0

    def render(self):
        print("noop")