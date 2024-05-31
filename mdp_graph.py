from collections.abc import Hashable
from collections import defaultdict


class MDPGraph(object):
    def __init__(self):
        self.neighbors = defaultdict(set)  # Set to record neighbors
        self.s_a_ns_transition_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.s_a_ns_rewards = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def add_transition(self, state: Hashable, action: Hashable, next_state: Hashable, probability: float):
        assert 0.0 <= probability <= 1.0
        was_zero = self.s_a_ns_transition_probs[state][action][next_state] == 0.0
        self.s_a_ns_transition_probs[state][action][next_state] = probability

        if probability > 0.0:
            self.neighbors[state].add(next_state)
        elif probability == 0.0 and not any(self.s_a_ns_transition_probs[state][action][ns] > 0.0 for ns in
                                            self.s_a_ns_transition_probs[state][action]):
            self.neighbors[state].discard(next_state)

    def add_reward(self, state: Hashable, action: Hashable, next_state: Hashable, reward: float):
        if self.s_a_ns_transition_probs[state][action][next_state] > 0.0:
            self.s_a_ns_rewards[state][action][next_state] = reward

    def get_neighbors(self, state: Hashable) -> set[Hashable]:
        return self.neighbors[state]


class PolicyGraph(MDPGraph):
    pass


class OptimalPolicyGraph(MDPGraph):
    def __init__(self, gamma: float, threshold: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.values = defaultdict(float)
        self.optimal_policy_distributions = defaultdict(lambda: defaultdict(float))

    def load_graph(self, graph: MDPGraph):
        self.neighbors = graph.neighbors
        self.s_a_ns_transition_probs = graph.s_a_ns_transition_probs
        self.s_a_ns_rewards = graph.s_a_ns_rewards

    def value_iteration(self, max_iterations: int = 1000):
        for _ in range(max_iterations):
            delta = 0
            new_values = defaultdict(float)
            for state in self.s_a_ns_transition_probs:
                max_value = float('-inf')
                for action, next_states in self.s_a_ns_transition_probs[state].items():
                    action_value = 0
                    for next_state, probability in next_states.items():
                        reward = self.s_a_ns_rewards[state][action][next_state]
                        action_value += probability * (reward + self.gamma * self.values[next_state])
                    max_value = max(max_value, action_value)
                new_values[state] = max_value
                delta = max(delta, abs(new_values[state] - self.values[state]))
            self.values = new_values
            if delta < self.threshold:
                break

    def compute_optimal_policy(self):
        for state in self.s_a_ns_transition_probs:
            action_values = []
            for action, next_states in self.s_a_ns_transition_probs[state].items():
                action_value = 0
                for next_state, probability in next_states.items():
                    reward = self.s_a_ns_rewards[state][action][next_state]
                    action_value += probability * (reward + self.gamma * self.values[next_state])
                action_values.append((action, action_value))

            # Find the maximum value and consider actions within the threshold
            max_value = max(action_value for action, action_value in action_values)
            optimal_actions = [
                action for action, action_value in action_values
                if abs(action_value - max_value) < self.threshold
            ]

            # Distribute probability equally among optimal actions
            optimal_probability = 1.0 / len(optimal_actions)
            for action in optimal_actions:
                self.optimal_policy_distributions[state][action] = optimal_probability

    def get_optimal_policy(self):
        return self.optimal_policy_distributions
