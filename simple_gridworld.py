import collections
import random
from itertools import product

import gymnasium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygame
import torch
from gymnasium import spaces
from torchvision.transforms import transforms

from mdp_graph import MDPGraph, OptimalPolicyGraph

ACTIONS = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
}

deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
perpendicular_moves = {
    0: [(0, -1), (0, 1)],  # Action 0 (up) -> left or right
    1: [(0, -1), (0, 1)],  # Action 1 (down) -> left or right
    2: [(-1, 0), (1, 0)],  # Action 2 (left) -> up or down
    3: [(-1, 0), (1, 0)],  # Action 3 (right) -> up or down
}

def rotate_grid(grid, coords):
    """Rotate the grid randomly by 0, 90, 180, or 270 degrees. Update non-None coords accordingly, leaving None values in place.

    Args:
        grid (np.array): The grid to rotate.
        coords (list of tuple or None): List of (row, col) coordinates or None values to update.

    Returns:
        tuple: The rotated grid and the updated list of coordinates with None values preserved.
    """
    rotations = random.choice([0, 1, 2, 3])
    grid_height, grid_width = grid.shape
    new_coords = []
    for coord in coords:
        if coord is not None:
            row, col = coord
            if rotations == 1:  # 90 degrees
                new_coords.append((col, grid_height - 1 - row))
            elif rotations == 2:  # 180 degrees
                new_coords.append((grid_height - 1 - row, grid_width - 1 - col))
            elif rotations == 3:  # 270 degrees
                new_coords.append((grid_width - 1 - col, row))
            else:  # 0 degrees, no change
                new_coords.append(coord)
        else:
            new_coords.append(None)  # Preserve None in its position
    return np.rot90(grid, k=rotations), new_coords


def flip_grid(grid, coords):
    """Flip the grid randomly: horizontally, vertically, or not at all. Update non-None coords accordingly, leaving None values in place.

    Args:
        grid (np.array): The grid to flip.
        coords (list of tuple or None): List of (row, col) coordinates or None values to update.

    Returns:
        tuple: The flipped grid and the updated list of coordinates with None values preserved.
    """
    flip_type = random.choice(["horizontal", "vertical", "none"])
    grid_height, grid_width = grid.shape
    new_coords = []
    for coord in coords:
        if coord is not None:
            row, col = coord
            if flip_type == "horizontal":
                new_coords.append((row, grid_width - 1 - col))
            elif flip_type == "vertical":
                new_coords.append((grid_height - 1 - row, col))
            else:  # No flip, preserve original coord
                new_coords.append(coord)
        else:
            new_coords.append(None)  # Preserve None in its position
    if flip_type == "horizontal":
        return np.fliplr(grid), new_coords
    elif flip_type == "vertical":
        return np.flipud(grid), new_coords
    return grid, new_coords  # No flip performed, return original grid and coords with None preserved.


def get_traversed_grids(pos: np.ndarray, previous_position: np.ndarray):
    # Determine the ranges for each dimension
    ranges = [range(min(a, b), max(a, b) + 1) for a, b in zip(previous_position, pos)]

    # Generate all combinations of these ranges across dimensions
    all_points = list(product(*ranges))

    # Convert tuples back to lists (or to whatever format is preferred)
    traversed_grids = [list(point) for point in all_points]

    return set(tuple(grid) for grid in traversed_grids)


def handle_keyboard_input() -> int:
    action = None
    while action is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
    return action


class SimpleGridWorld(gymnasium.Env, collections.abc.Iterator):
    """

    The TextGridWorld class is an environment class that represents a grid world. The grid world is loaded from a text file, where each character represents a cell in the grid. The grid
    * can contain walls ('W'), traps ('T'), the agent ('A'), and the goal ('G').

    This class inherits from the 'gymnasium.Env' class and overrides its methods.

    Args:
        text_file (str): The path to the text file containing the grid. Each line in the file represents a row in the grid, and each character in the line represents a cell in the row.
        cell_size (tuple, optional): The size of each grid cell in pixels. Defaults to (20, 20).
        agent_position (tuple, optional): The initial position of the agent in the grid. If not specified, a random position will be assigned. Defaults to None.
        goal_position (tuple, optional): The position of the goal in the grid. If not specified, a random position will be assigned. Defaults to None.

    Attributes:
        metadata (dict): Metadata about the environment, including the supported render modes.
        grid (numpy array): The grid representing the environment.
        action_space (gymnasium.Space): The action space for the agent.
        observation_space (gymnasium.Space): The observation space for the agent.
        cell_size (tuple): The size of each grid cell in pixels.
        screen_size (tuple): The size of the screen in pixels.
        viewer (pygame.Surface): The surface used for rendering the environment.
        cached_surface (pygame.Surface): The cached surface used for rendering.
        _agent_position (tuple): The initial position of the agent, might be None.
        _goal_position (tuple): The position of the goal, might be None.
        agent_position (tuple): The current position of the agent.
        goal_position (tuple): The current position of the goal.

    Methods:
        load_grid(text_file): Loads the grid from a text file.
        reset_positions(): Resets the positions of the agent and the goal.
        assign_position(occupied_positions): Assigns a position that is not occupied by the agent or the goal.
        step(action): Performs a step in the environment given an action.
        reset(seed=None, options=None): Resets the environment to its initial state.
        get_observation(): Returns the current observation of the environment.
        _render_to_surface(): Renders the environment to the cached surface.
        render(mode='human'): Renders the environment in a specified mode.
        close(): Closes the environment.

    """
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self, text_file, cell_size=(20, 20), obs_size=(128, 128), agent_position=None, goal_position=None,
                 random_traps=0, make_random=False, max_steps=128, trans_prob=1.0
                 # compute_optimal=False,
                 ):
        super(SimpleGridWorld, self).__init__()
        self.random = make_random
        self.max_steps = max_steps
        self.step_count = 0
        self.trans_prob = trans_prob

        self.grid = self.load_grid(text_file)
        self.cell_size = cell_size
        self.screen_size = (self.grid.shape[1] * cell_size[0], self.grid.shape[0] * cell_size[1])
        self.viewer = None
        self.cached_surface = None
        self.window = None

        self._agent_position = agent_position
        self._goal_position = goal_position

        self.num_random_traps = random_traps
        self.pos_random_traps = []

        self.agent_position = None
        self.goal_position = None

        # gymnasium required
        self.obs_size = obs_size  # H, W
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, obs_size[0], obs_size[1]), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.num_actions = len(ACTIONS)

        pygame.init()

        self.occupied_positions = set()
        self.reset_positions()

        self.shape = self.grid.shape
        self.iter_index = 0
        self.iter_coord = (0, 0)

        self._traj = []

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __iter__(self):
        return self

    def __next__(self):
        temp_agent_position = self.agent_position  # should be helpful for not messing up the coordinates
        if self.iter_index >= len(self):
            raise StopIteration
        row, col = self.iter_coord
        self.agent_position = self.iter_coord
        temp_iter_coord = self.iter_coord
        col += 1
        if col >= self.shape[1]:
            col = 0
            row += 1
        self.iter_coord = (row, col)
        self.iter_index += 1

        if self.grid[self.agent_position[0], self.agent_position[1]] not in ['W']:
            terminated = self.agent_position == self.goal_position  # or self.grid[self.agent_position] == 'X'
            self._render_to_surface()
            observation = self.get_observation()
            observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
            observation /= 255.0 if observation.max() > 1.0 else 1.0

            connections = {}
            rewards = {}
            deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            for action in ACTIONS.keys():
                delta = deltas[action]
                new_position = (self.agent_position[0] + delta[0], self.agent_position[1] + delta[1])
                rewards[len(ACTIONS)] = 5 if self.agent_position == self.goal_position else -1 if (self.grid[
                                                                                                       self.agent_position] == 'X' or self.agent_position in self.pos_random_traps) else -0.01
                if 0 <= new_position[0] < self.grid.shape[0] and 0 <= new_position[1] < self.grid.shape[1]:
                    if self.grid[new_position] not in ['W']:
                        # new position reachable
                        connections[action] = 1.0
                        rewards[action] = 0.0
                        rewards[action] += 5 if new_position == self.goal_position else -1 if (
                                self.grid[new_position] == 'X' or new_position in self.pos_random_traps) else -0.01
                    elif self.grid[new_position] == 'W':
                        # hits wall
                        connections[action] = 0.0
                        rewards[action] = -0.1
                        rewards[action] += rewards[len(ACTIONS)]
            connections = torch.tensor([connections[i] for i in range(len(connections))])
            rewards = torch.tensor([rewards[i] for i in range(len(rewards))])
            self.agent_position = temp_agent_position
            return observation, terminated, temp_iter_coord, connections, rewards, self.optimal_policy[temp_iter_coord]

        self.agent_position = temp_agent_position
        return None, None, temp_iter_coord, None, None, self.optimal_policy[temp_iter_coord]

    def iter_reset(self):
        self.iter_index = 0
        self.iter_coord = (0, 0)

    def directed_reachable_positions(self, position: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = position
        potential_neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]  # Neighbors are up, down, left, right
        reachable_positions = []
        if self.grid[x, y] in ['W'] or position == self.goal_position:
            return reachable_positions
        for potential_neighbour in potential_neighbours:
            if 0 <= potential_neighbour[0] < self.grid.shape[0] and 0 <= potential_neighbour[1] < self.grid.shape[1]:
                if self.grid[potential_neighbour[0], potential_neighbour[1]] not in ['W']:
                    reachable_positions.append(potential_neighbour)
        return reachable_positions

    def make_directed_graph(self, filepath: None or str = None, show=False):
        graph = nx.DiGraph()
        node_colors = {}  # Dictionary to store colors keyed by node

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                node = (x, y)
                if self.grid[x, y] not in ['W']:
                    graph.add_node(node)
                    for neighbor in self.directed_reachable_positions(node):
                        graph.add_edge(node, neighbor)

                # Assign colors based on conditions, directly using the node as a key
                if self.grid[x, y] in ['X'] or node in self.pos_random_traps:
                    node_colors[node] = 'red'  # Trap position
                elif node == self.goal_position or node == self._goal_position:
                    node_colors[node] = 'green'  # Goal position
                else:
                    node_colors[node] = 'blue'  # Default color for other nodes

        if filepath is not None:
            # Apply the colors when drawing
            colors = [node_colors.get(node, 'blue') for node in graph.nodes()]
            pos = nx.kamada_kawai_layout(graph)  # Positions for all nodes
            node_sizes = [100 for n in graph.nodes()]
            nx.draw(graph, pos, with_labels=True, arrows=True, node_color=colors, node_size=node_sizes, font_size=8, )
            # plt.show()
            plt.savefig(filepath, dpi=600)  # Set the resolution with the `dpi` argument
            plt.close()

        if show:
            # Apply the colors when drawing
            colors = [node_colors.get(node, 'blue') for node in graph.nodes()]
            pos = nx.kamada_kawai_layout(graph)  # Positions for all nodes
            node_sizes = [100 for n in graph.nodes()]
            nx.draw(graph, pos, with_labels=True, arrows=True, node_color=colors, node_size=node_sizes, font_size=8, )
            plt.show()

        return graph

    def make_mdp_graph(self):
        mdp_graph = MDPGraph()
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                state = (row, col)
                if self.grid[state] in ['W'] or state == self.goal_position:
                    continue
                for action in range(len(ACTIONS)):
                    forward_delta = deltas[action]
                    perpendicular_deltas = perpendicular_moves[action]
                    forward_prob = self.trans_prob
                    perpendicular_prob = 0.5 * (1.0 - self.trans_prob)
                    _deltas = [forward_delta, *perpendicular_deltas,]
                    _probs = [forward_prob, perpendicular_prob, perpendicular_prob]

                    for delta, prob in zip(_deltas, _probs):
                        next_state = (state[0] + delta[0], state[1] + delta[1])
                        rwd = -0.01
                        if self.grid[next_state] == 'W':
                            next_state = state
                            rwd -= 0.1
                        elif self.grid[next_state] == 'X' or next_state in self.pos_random_traps:
                            rwd -= 1.0
                        elif next_state == self.goal_position:
                            rwd = 5.0
                        if prob > 0.0:
                            mdp_graph.add_transition(state, action, next_state, prob)
                            mdp_graph.add_reward(state, action, next_state, rwd)
        return mdp_graph

    def load_grid(self, text_file):
        with open(text_file, 'r') as file:
            lines = file.read().splitlines()
        self.grid = np.array([list(line) for line in lines])
        return self.grid

    def reset_positions(self, agent_only=False):
        self.occupied_positions = set()

        self.agent_position = self._agent_position if self._agent_position else self.assign_position()
        self.occupied_positions.add(self.agent_position)

        if agent_only:
            return

        self.goal_position = self._goal_position if self._goal_position else self.assign_position()
        self.occupied_positions.add(self.goal_position)

        self.pos_random_traps = []
        for i in range(random.randint(0, self.num_random_traps)):
            try:
                trap = self.assign_position()
                self.pos_random_traps.append(trap)
                self.occupied_positions.add(trap)
            except RuntimeError:
                print("Warning: Not enough empty position assignable for random traps.")

    def assign_position(self):
        counter = 0
        while True:
            position = (random.randint(0, self.grid.shape[0] - 1), random.randint(0, self.grid.shape[1] - 1))
            if position not in self.occupied_positions and self.grid[position] not in ['W', 'X', 'G']:
                return position
            counter += 1
            if counter >= 16384:
                raise RuntimeError("Cannot assign any more positions.")

    def step(self, action):
        self.step_count += 1

        if random.random() < self.trans_prob:
            delta = deltas[action]
        else:
            delta = random.choice(perpendicular_moves[action])

        new_position = (self.agent_position[0] + delta[0], self.agent_position[1] + delta[1])

        hits_wall = False
        if 0 <= new_position[0] < self.grid.shape[0] and 0 <= new_position[1] < self.grid.shape[1]:
            if self.grid[new_position] not in ['W']:
                self.agent_position = new_position
            elif self.grid[new_position] == 'W':
                hits_wall = True

        terminated = self.agent_position == self.goal_position  # or self.grid[self.agent_position] == 'X'
        truncated = False
        reward = 5 if self.agent_position == self.goal_position else -1 if (
                self.grid[self.agent_position] == 'X' or self.agent_position in self.pos_random_traps) else -0.01
        if hits_wall:
            reward -= 0.1

        if self.grid[self.agent_position] in ['W']:
            print('traj:', self._traj)
            print(len(self._traj))
            print(self.step_count, self.max_steps)
            print(self.grid)
            print(self.agent_position, self.grid[self.agent_position])
            raise AssertionError

        self._render_to_surface()
        observation = self.get_observation()
        observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
        observation /= 255.0 if observation.max() > 1.0 else 1.0
        if self.step_count >= self.max_steps:
            terminated = True
            truncated = True
            reward = 0

        self._traj.append(action)
        self._traj.append(self.agent_position)
        return observation, reward, terminated, truncated, {
            "position": self.agent_position,
            "goal_position": self.goal_position,
        }

    def reset(self, seed=None, options=None, no_random=False):
        super().reset(seed=seed)
        self.step_count = 0
        if not no_random:
            if self.random:
                # Randomly rotate the grid
                self.grid, (self._agent_position, self._goal_position, self.agent_position, self.goal_position) \
                    = rotate_grid(self.grid,
                                  [self._agent_position, self._goal_position, self.agent_position, self.goal_position])
                # Randomly flip the grid
                self.grid, (self._agent_position, self._goal_position, self.agent_position, self.goal_position) \
                    = flip_grid(self.grid,
                                [self._agent_position, self._goal_position, self.agent_position, self.goal_position])
            self.reset_positions()
        else:
            self.reset_positions(agent_only=True)

        assert self.grid[self.agent_position] not in ['W', 'X', 'G']
        self._traj = []
        self._traj.append(self.agent_position)

        self._render_to_surface()
        observation = self.get_observation()
        observation = torch.tensor(observation).permute(2, 0, 1).type(torch.float32)
        observation /= 255.0 if observation.max() > 1.0 else 1.0

        return observation, {"position": self.agent_position, "goal_position": self.goal_position, }

    def get_observation(self):
        observation = pygame.surfarray.array3d(self.cached_surface)
        observation = np.transpose(observation, (1, 0, 2))  # WHC - HWC
        return observation

    def _render_to_surface(self):
        if self.viewer is None:
            self.viewer = pygame.Surface(self.screen_size)
            self.cached_surface = pygame.Surface(self.obs_size)  # 使用obs_size代替screen_size

        self.viewer.fill((255, 255, 255))

        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                cell_content = self.grid[y, x]
                rect = pygame.Rect(x * self.cell_size[0], y * self.cell_size[1], self.cell_size[0], self.cell_size[1])
                if cell_content == 'W':
                    pygame.draw.rect(self.viewer, (0, 0, 0), rect)
                elif cell_content == 'X' or (y, x) in self.pos_random_traps:
                    pygame.draw.rect(self.viewer, (255, 0, 0), rect)

        goal_rect = pygame.Rect(self.goal_position[1] * self.cell_size[0], self.goal_position[0] * self.cell_size[1],
                                self.cell_size[0], self.cell_size[1])
        pygame.draw.rect(self.viewer, (0, 255, 0), goal_rect)

        agent_center = (int(self.agent_position[1] * self.cell_size[0] + self.cell_size[0] / 2),
                        int(self.agent_position[0] * self.cell_size[1] + self.cell_size[1] / 2))
        pygame.draw.circle(self.viewer, (0, 0, 255), agent_center, int(self.cell_size[0] / 2))

        scaled_surface = pygame.transform.scale(self.viewer, self.obs_size)
        self.cached_surface.blit(scaled_surface, (0, 0))

    def render(self, mode='rgb_array'):
        if mode == 'human':
            if self.window is None:
                self.window = pygame.display.set_mode(self.screen_size)

            if self.cached_surface is not None:
                self.window.blit(self.cached_surface, (0, 0))
                pygame.display.flip()
        elif mode == 'rgb_array':
            return self.get_observation()
        elif mode == 'console':
            for y in range(self.grid.shape[0]):
                row = ''
                for x in range(self.grid.shape[1]):
                    if (y, x) == self.agent_position:
                        row += 'A'
                    elif (y, x) == self.goal_position:
                        row += 'G'
                    elif self.grid[y, x] == 'W':
                        row += 'W'
                    elif self.grid[y, x] == 'X':
                        row += 'X'
                    else:
                        row += ' '
                print(row)

    def close(self):
        if self.viewer is not None:
            pygame.quit()
            self.viewer = None

    def action_transition_and_reward(self, coord: tuple[int, int]) -> tuple[tuple[tuple[int, int], float], ...]:
        # Define deltas for 4 actions: up, down, left, right
        actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        transitions: list[tuple[tuple[int, int], float]] = []

        for action, delta in actions.items():
            # Compute new position based on the action
            new_position = (coord[0] + delta[0], coord[1] + delta[1])
            if 0 <= new_position[0] < self.grid.shape[0] and 0 <= new_position[1] < self.grid.shape[1]:
                # Check for walls and special areas
                if self.grid[new_position] == 'W':
                    # If it's a wall, stay in place, apply wall collision penalty
                    reward = -0.1
                    transitions.append((coord, reward))
                else:
                    # If it's not a wall, check for goal and traps
                    if new_position == self.goal_position:
                        reward = 5.0
                    elif new_position in self.pos_random_traps or self.grid[new_position] == 'X':
                        reward = -1.0
                    else:
                        reward = -0.01
                    transitions.append((new_position, reward))
            else:
                # If out of bounds, stay in place, no extra penalty
                transitions.append((coord, -0.01))

        return tuple(transitions)


def preprocess_image(img: np.ndarray, rotate=False, size=None) -> torch.Tensor:
    # Convert the numpy array to a PIL Image
    transform_to_pil = transforms.ToPILImage()
    pil_image = transform_to_pil(img)
    # Initialize the transformation list
    transformations = []
    # Randomly rotate the image
    if rotate:
        rotation_degrees = np.random.choice([0, 90, 180, 270])
        transformations.append(transforms.RandomRotation([rotation_degrees, rotation_degrees]))
    # Resize the image if size is specified
    if size is not None:
        transformations.append(transforms.Resize(size))
    # Convert the PIL Image back to a tensor
    transformations.append(transforms.ToTensor())
    # Compose all transformations
    transform_compose = transforms.Compose(transformations)
    # Apply transformations
    processed_tensor = transform_compose(pil_image)
    # Normalize the tensor to [0, 1] (if not already normalized)
    processed_tensor /= 255.0 if processed_tensor.max() > 1.0 else 1.0
    # Add a batch dimensionget_optimal_policy_str
    processed_tensor = processed_tensor.unsqueeze(0)
    return processed_tensor


if __name__ == "__main__":
    env = SimpleGridWorld('maps/simple_grid/gridworld-maze-13.txt', make_random=True, random_traps=0,
                          agent_position=None, goal_position=(1, 8), max_steps=1024, trans_prob=0.8)
    obs, info = env.reset()
    mdp_graph = env.make_mdp_graph()
    optimal_policy_graph = OptimalPolicyGraph()
    optimal_policy_graph.load_graph(mdp_graph)
    optimal_policy_graph.value_iteration(0.99)
    optimal_policy_graph.compute_optimal_policy(0.99)
    optimal_policy = optimal_policy_graph.get_optimal_policy()
    optimal_policy_graph.probability_iteration()
    probability_distribution = optimal_policy_graph.get_probability_distribution()
    pass
