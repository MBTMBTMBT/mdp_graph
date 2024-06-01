from collections.abc import Hashable
from collections import defaultdict


class MDPGraph(object):
    def __init__(self):
        self.state_neighbors = defaultdict(set)  # Set to record neighbors
        self.state_neighbors_inverse = defaultdict(set)  # Set to record inverse neighbors
        self.state_actions = defaultdict(set)  # Set to record possible actions for each state
        self.s_a_ns_transition_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.s_a_ns_rewards = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def load_graph(self, graph: 'MDPGraph'):
        self.state_neighbors = graph.state_neighbors
        self.state_neighbors_inverse = graph.state_neighbors_inverse
        self.state_actions = graph.state_actions
        self.s_a_ns_transition_probs = graph.s_a_ns_transition_probs
        self.s_a_ns_rewards = graph.s_a_ns_rewards

    def add_transition(self, state: Hashable, action: Hashable, next_state: Hashable, probability: float):
        assert 0.0 <= probability <= 1.0
        current_prob = self.s_a_ns_transition_probs[state][action][next_state]
        self.s_a_ns_transition_probs[state][action][next_state] = probability

        if probability > 0.0:
            self.state_neighbors[state].add(next_state)
            self.state_neighbors_inverse[next_state].add(state)
            self.state_actions[state].add(action)
        elif current_prob > 0.0 and probability == 0.0:
            # Remove the next_state from neighbors if its probability is 0
            if all(self.s_a_ns_transition_probs[state][action][ns] == 0.0 for ns in self.s_a_ns_transition_probs[state][action]):
                self.state_neighbors[state].discard(next_state)
                self.state_actions[state].discard(action)

            # Clean up empty actions
            if not any(self.s_a_ns_transition_probs[state][action][ns] > 0.0 for ns in self.s_a_ns_transition_probs[state][action]):
                del self.s_a_ns_transition_probs[state][action]

            # Remove state from inverse neighbors if no transitions to next_state exist
            if not any(self.s_a_ns_transition_probs[other_state][other_action][next_state] > 0.0
                       for other_state in self.s_a_ns_transition_probs
                       for other_action in self.s_a_ns_transition_probs[other_state]):
                self.state_neighbors_inverse[next_state].discard(state)

    def add_reward(self, state: Hashable, action: Hashable, next_state: Hashable, reward: float):
        if self.s_a_ns_transition_probs[state][action][next_state] > 0.0:
            self.s_a_ns_rewards[state][action][next_state] = reward

    def get_neighbors(self, state: Hashable) -> set[Hashable]:
        return self.state_neighbors[state]

    def get_inverse_neighbors(self, state: Hashable) -> set[Hashable]:
        return self.state_neighbors_inverse[state]


class PolicyGraph(MDPGraph):
    def __init__(self,):
        super().__init__()
        self.policy_distributions = defaultdict(lambda: defaultdict(float))
        self.state_probabilities = defaultdict(float)

    def uniform_policy(self):
        for state in self.state_actions.keys():
            prob = 1.0 / len(self.state_actions[state])
            for action in self.state_actions[state]:
                self.policy_distributions[state][action] = prob

    def probability_iteration(self, threshold: float = 1e-4, max_iterations: int = 1e5):
        for _ in range(max_iterations):
            delta = 0
            new_probs = defaultdict(float)
            for state in self.s_a_ns_transition_probs:
                for action in self.state_actions[state]:
                    pass


class OptimalPolicyGraph(PolicyGraph):
    def __init__(self,):
        super().__init__()
        self.values = defaultdict(float)

    def value_iteration(self, gamma: float, threshold: float = 1e-4, max_iterations: int = 1e5):
        for _ in range(max_iterations):
            delta = 0
            new_values = defaultdict(float)
            for state in self.s_a_ns_transition_probs:
                max_value = float('-inf')
                for action, next_states in self.s_a_ns_transition_probs[state].items():
                    action_value = 0
                    for next_state, probability in next_states.items():
                        reward = self.s_a_ns_rewards[state][action][next_state]
                        action_value += probability * (reward + gamma * self.values[next_state])
                    max_value = max(max_value, action_value)
                new_values[state] = max_value
                delta = max(delta, abs(new_values[state] - self.values[state]))
            self.values = new_values
            if delta < threshold:
                break

    def compute_optimal_policy(self, gamma: float, threshold: float = 1e-4,):
        for state in self.s_a_ns_transition_probs:
            action_values = []
            for action, next_states in self.s_a_ns_transition_probs[state].items():
                action_value = 0
                for next_state, probability in next_states.items():
                    reward = self.s_a_ns_rewards[state][action][next_state]
                    action_value += probability * (reward + gamma * self.values[next_state])
                action_values.append((action, action_value))

            # Find the maximum value and consider actions within the threshold
            max_value = max(action_value for action, action_value in action_values)
            optimal_actions = [
                action for action, action_value in action_values
                if abs(action_value - max_value) < threshold
            ]

            # Distribute probability equally among optimal actions
            optimal_probability = 1.0 / len(optimal_actions)
            for action in optimal_actions:
                self.policy_distributions[state][action] = optimal_probability

    def get_optimal_policy(self):
        return self.policy_distributions
