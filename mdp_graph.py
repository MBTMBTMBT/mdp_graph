import math
from collections import defaultdict
from collections.abc import Hashable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def grid_layout(g: nx.Graph) -> dict[Hashable, tuple[float, float]]:
    '''
    Can only be used for gridworlds.
    :param g: graph to visualize.
    :return: networkx layout.
    '''
    pos = {}
    for node in g.nodes():
        # Assuming the node is a tuple (row, col) representing its position in the grid
        row, col = node
        pos[node] = (col, -row)  # Use (col, -row) to maintain vertical alignment
    return pos


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
            if all(self.s_a_ns_transition_probs[state][action][ns] == 0.0 for ns in
                   self.s_a_ns_transition_probs[state][action]):
                self.state_neighbors[state].discard(next_state)
                self.state_actions[state].discard(action)

            # Clean up empty actions
            if not any(self.s_a_ns_transition_probs[state][action][ns] > 0.0 for ns in
                       self.s_a_ns_transition_probs[state][action]):
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

    def visualize(self, title="MDP State Transition Graph", highlight_states: set or None = None, figsize=(5, 5), dpi=90, node_size=400, node_font_size=8,
                  arrowsize=10, use_grid_layout=True):
        # Create a directed graph
        g = nx.DiGraph()

        # Add edges for the transitions
        for state, neighbors in self.state_neighbors.items():
            for neighbor in neighbors:
                if state != neighbor:
                    g.add_edge(state, neighbor)

        if use_grid_layout:
            pos = grid_layout(g)
        else:
            # Use kamada_kawai_layout for better spacing
            pos = nx.kamada_kawai_layout(g)

            # Adjusting the position for more spacing
            for key in pos:
                pos[key] *= 3.5  # Increase spacing by multiplying the positions

        # Determine node colors
        if highlight_states:
            node_colors = ['red' if node in highlight_states else 'skyblue' for node in g.nodes()]
        else:
            node_colors = ['skyblue' for node in g.nodes()]

        plt.figure(figsize=figsize, dpi=dpi)
        nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color=node_colors)
        nx.draw_networkx_edges(g, pos, arrowstyle='-|>', arrowsize=arrowsize, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(g, pos, font_size=node_font_size, font_weight="bold")

        plt.title(title)
        plt.show()


class PolicyGraph(MDPGraph):
    def __init__(self, ):
        super().__init__()
        self.policy_distributions = defaultdict(lambda: defaultdict(float))
        self.state_probabilities = defaultdict(float)
        self.control_info = defaultdict(float)

    def uniform_policy(self):
        for state in self.state_actions.keys():
            prob = 1.0 / len(self.state_actions[state])
            for action in self.state_actions[state]:
                self.policy_distributions[state][action] = prob

    def probability_iteration(self, threshold: float = 1e-5, max_iterations: int = int(1e5)):
        for state in self.s_a_ns_transition_probs:
            self.state_probabilities[state] = 1.0 / len(self.s_a_ns_transition_probs)
        for _ in range(max_iterations):
            delta = 0
            new_probs = defaultdict(float)
            for state in self.s_a_ns_transition_probs:
                state_prob = 0.0
                for inverse_neighbour in self.get_inverse_neighbors(state):
                    for action in self.state_actions[inverse_neighbour]:
                        state_prob += self.state_probabilities[inverse_neighbour] * \
                                      self.policy_distributions[inverse_neighbour][action] * \
                                      self.s_a_ns_transition_probs[inverse_neighbour][action][state]
                new_probs[state] = state_prob
                delta = max(delta, abs(new_probs[state] - self.state_probabilities[state]))

            total_prob = sum(new_probs.values())
            if total_prob > 0.0:
                for state in new_probs:
                    new_probs[state] /= total_prob

            self.state_probabilities = new_probs
            if delta < threshold:
                break

    def get_probability_distribution(self):
        return self.state_probabilities

    def control_info_iteration(self, gamma: float, threshold: float = 1e-3, max_iterations: int = int(1e5)):
        for _ in range(max_iterations):
            delta = 0
            new_control_info = defaultdict(float)
            for state in self.s_a_ns_transition_probs:
                state_control_info = 0.0
                delta_control_info = 0.0
                for action in self.state_actions[state]:
                    action_prob = self.policy_distributions[state][action]
                    if action_prob <= 0.0:
                        continue
                    delta_control_info += action_prob * math.log2(action_prob / (1.0 / len(self.state_actions[state])))
                for action, next_states in self.s_a_ns_transition_probs[state].items():
                    for next_state, trans_prob in next_states.items():
                        neighbour_info = self.control_info[next_state]
                        action_prob = self.policy_distributions[state][action]
                        state_control_info += action_prob * trans_prob * neighbour_info
                state_control_info *= gamma
                new_control_info[state] = delta_control_info + state_control_info
                delta = max(delta, abs(new_control_info[state] - self.control_info[state]))
            self.control_info = new_control_info
            if delta < threshold:
                break

    def get_control_info(self):
        return self.control_info

    def visualize_policy_and_control_info(self, title="Policy and Control Info", highlight_states: set or None = None,
                                          figsize=(5, 5), dpi=90,
                                          node_size=400, node_font_size=8, arrow_size=10, arrow_font_size=4,
                                          use_grid_layout=True):
        # Create a directed graph
        g = nx.DiGraph()

        # Add edges based on policy distributions
        for state in self.policy_distributions:
            for action, prob in self.policy_distributions[state].items():
                if prob > 0:
                    # Find the next state with the highest transition probability
                    next_state = max(self.s_a_ns_transition_probs[state][action],
                                     key=self.s_a_ns_transition_probs[state][action].get)
                    g.add_edge(state, next_state, action=action, prob=prob)

        if use_grid_layout:
            pos = grid_layout(g)
        else:
            # Use kamada_kawai_layout for better spacing
            pos = nx.kamada_kawai_layout(g)

            # Adjusting the position for more spacing
            for key in pos:
                pos[key] *= 3.5  # Increase spacing by multiplying the positions

        # Determine node colors
        if highlight_states:
            node_colors = ['red' if node in highlight_states else 'skyblue' for node in g.nodes()]
        else:
            node_colors = ['skyblue' for node in g.nodes()]

        plt.figure(figsize=figsize, dpi=dpi)
        nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color=node_colors)
        nx.draw_networkx_edges(g, pos, arrowstyle='-|>', arrowsize=arrow_size, connectionstyle='arc3,rad=0.1')

        # Draw node labels (control information)
        node_labels = {state: f'{state}\n{self.control_info[state]:.1f}' for state in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=node_font_size, font_weight="bold")

        # Draw edge labels (action and probability)
        edge_labels = {(u, v): f'{g[u][v]["prob"]:.1f}' for u, v in g.edges()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=arrow_font_size)

        plt.title(title)
        plt.show()

    def draw_action_distribution(self, num_cols: int = 5, figsize=(20, 4), dpi=90):
        states = sorted(list(self.state_actions.keys()))
        actions_set = {action for actions in self.state_actions.values() for action in actions}
        actions = sorted(actions_set)
        num_states = len(states)
        num_actions = len(actions)
        states_per_col = int(np.ceil(num_states / num_cols))

        fig, axs = plt.subplots(1, num_cols, figsize=figsize, constrained_layout=True, dpi=dpi)

        for i in range(num_cols):
            start_idx = i * states_per_col
            end_idx = min(start_idx + states_per_col, num_states)
            grid = np.zeros((end_idx - start_idx, num_actions))

            for j, state in enumerate(states[start_idx:end_idx]):
                for k, action in enumerate(actions):
                    grid[j, k] = self.policy_distributions[state].get(action, 0)

            ax = axs[i]
            cax = ax.imshow(grid, cmap='binary', aspect='auto', vmin=0, vmax=1)

            # Set action labels for the top of each column
            ax.set_xticks(np.arange(num_actions))
            ax.set_xticklabels(actions)

            # Set state labels for each row
            ax.set_yticks(np.arange(end_idx - start_idx))
            ax.set_yticklabels(states[start_idx:end_idx])

            ax.set_title(f'Part {i + 1}')

        plt.colorbar(cax, ax=axs, orientation='vertical', fraction=.01)
        plt.suptitle('Action Distribution for States')
        plt.show()

    def compute_policy_relevant_info(self, state_set: set or None = None):
        if state_set is None:
            state_set = set(self.state_actions.keys())

        state_prob_prior = 1.0 / len(state_set)

        policy_entropy = 0.0
        actions_set = {action for actions in self.state_actions.values() for action in actions}
        action_probs = {action: 0.0 for action in actions_set}
        for action in actions_set:
            for state in state_set:
                action_probs[action] += self.policy_distributions[state][action] * state_prob_prior
            if action_probs[action] <= 0.0:
                continue
            policy_entropy -= action_probs[action] * math.log2(action_probs[action])

        dependent_policy_entropy = 0.0
        for state in state_set:
            for action in self.state_actions[state]:
                if self.policy_distributions[state][action] <= 0.0:
                    continue
                dependent_policy_entropy -= self.policy_distributions[state][action] * math.log2(self.policy_distributions[state][action])
                pass
        dependent_policy_entropy *= state_prob_prior

        policy_relevant_info = policy_entropy - dependent_policy_entropy

        return policy_relevant_info


class OptimalPolicyGraph(PolicyGraph):
    def __init__(self, ):
        super().__init__()
        self.values = defaultdict(float)

    def value_iteration(self, gamma: float, threshold: float = 1e-5, max_iterations: int = int(1e5)):
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

    def compute_optimal_policy(self, gamma: float, threshold: float = 1e-2, ):
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
