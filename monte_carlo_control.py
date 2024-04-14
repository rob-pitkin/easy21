# Author: Rob Pitkin
# Date: 4/14/2024
# Description: This file contains the Monte Carlo control logic for the game.

from game import Easy21, GameState, Card
from typing import Dict, List
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class MCControl:
    def __init__(self, n_0=100):
        initial_q_values = {}
        for dealer_card in range(1, 11):
            for player_sum in range(1, 22):
                for action in ["hit", "stick"]:
                    initial_q_values[
                        (GameState(Card(dealer_card, "black"), player_sum), action)
                    ] = 0
        self.q_values: Dict[tuple[GameState, str], float] = (
            initial_q_values  # State-action value function
        )
        initial_n_state = {}
        for dealer_card in range(1, 11):
            for player_sum in range(1, 22):
                initial_n_state[GameState(Card(dealer_card, "black"), player_sum)] = 0
        self.n_state: Dict[GameState, int] = (
            initial_n_state  # Number of times a state has been visited
        )
        initial_n_state_action = {}
        for dealer_card in range(1, 11):
            for player_sum in range(1, 22):
                for action in ["hit", "stick"]:
                    initial_n_state_action[
                        (GameState(Card(dealer_card, "black"), player_sum), action)
                    ] = 0
        self.n_state_action: Dict[tuple[GameState, str], int] = (
            initial_n_state_action  # Number of times a state-action pair has been visited
        )
        self.n_0 = n_0  # Epsilon constant for epsilon-greedy policy

    def get_epsilon(self, state: GameState) -> float:
        """
        Get the epsilon value for the given state

        Args:
            state (GameState): The state to get the epsilon value for

        Returns:
            float: The epsilon value for the given state
        """
        # Get the epsilon value for the given state
        return self.n_0 / (self.n_0 + self.n_state[state])

    def get_alpha(self, state: GameState, action: str) -> float:
        """
        Get the alpha value for the given state-action pair

        Args:
            state (GameState): The state to get the alpha value for
            action (str): The action to get the alpha value for

        Returns:
            float: The alpha value for the given state-action pair
        """
        # Get the alpha value for the given state-action pair
        return 1 / self.n_state_action[(state, action)]

    def get_q_value(self, state: GameState, action: str) -> float:
        """
        Get the Q-value for the given state-action pair

        Args:
            state (GameState): The state to get the Q-value for
            action (str): The action to get the Q-value for

        Returns:
            float: The Q-value for the given state-action pair
        """
        # Get the Q-value for the given state-action pair
        return self.q_values[(state, action)]

    def get_e_greedy_action(self, state: GameState) -> str:
        """
        Get the best action for the given state using an epsilon-greedy policy

        Args:
            state (GameState): The state to get the best action for

        Returns:
            str: The best action for the given state
        """
        # Get the action with the highest Q-value
        max_q = -float("inf")
        best_action = None
        random_number = random.random()
        if random_number > self.get_epsilon(state):
            for action in ["hit", "stick"]:
                q = self.get_q_value(state, action)
                if q > max_q:
                    max_q = q
                    best_action = action
        else:
            best_action = random.choice(["hit", "stick"])
        return best_action

    def sample_episode(self) -> List[tuple[GameState, str, int]]:
        """
        Sample an episode from the environment

        Returns:
            List[tuple[GameState, str, int]]: The episode sampled from the environment
        """
        # Sample an episode
        game = Easy21()
        episode = []
        cumulative_reward = 0
        while not game.is_finished:
            state = game.get_state()
            action = self.get_e_greedy_action(state)
            _, reward = game.step(state, action)
            cumulative_reward += reward
            episode.append((state, action, reward))
        return episode, cumulative_reward

    def update_q_values(self, episode: List[tuple[GameState, str, int]]) -> None:
        """
        Update the Q-values for the episode

        Args:
            episode (List[tuple[GameState, str, int]]): The episode to update the Q-values for

        Returns:
            None
        """
        # Update the Q-values for the episode
        G_t = 0
        # Iterate through the episode in reverse
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            G_t += reward
            self.n_state[state] = self.n_state[state]
            self.n_state_action[(state, action)] = (
                self.n_state_action[(state, action)] + 1
            )
            alpha = self.get_alpha(state, action)
            q = self.get_q_value(state, action)
            self.q_values[(state, action)] = q + alpha * (G_t - q)

    def train(self, n_episodes=1000) -> None:
        """
        Train the agent using Monte Carlo control

        Args:
            n_episodes (int): The number of episodes to train the agent for

        Returns:
            None
        """
        rewards = []
        # Train the agent
        for i in range(n_episodes):
            print(f"Training episode {i + 1}/{n_episodes}")
            episode, cumulative_reward = self.sample_episode()
            rewards.append(cumulative_reward)
            self.update_q_values(episode)
        # self.plot_rewards(rewards)
        self.plot_optimal_value_function()

    def plot_rewards(self, rewards: List[int]) -> None:
        """
        Plot the rewards

        Args:
            rewards (List[int]): The rewards to plot

        Returns:
            None
        """
        # Plot the rewards
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward vs Episode")
        plt.show()

    def plot_optimal_value_function(self) -> None:
        """
        Plot the optimal value function

        Returns:
            None
        """
        # Plot the optimal value function for black
        player_sum = np.arange(1, 22)
        dealer_showing = np.arange(1, 11)
        value_function = np.zeros((10, 21))
        for i in range(10):
            for j in range(21):
                state = GameState(Card(dealer_showing[i], "black"), player_sum[j])
                value_function[i, j] = max(
                    self.get_q_value(state, "hit"), self.get_q_value(state, "stick")
                )

        # Create 2D grids from player_sum and dealer_showing
        player_sum_grid, dealer_showing_grid = np.meshgrid(player_sum, dealer_showing)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the surface
        ax.plot_surface(
            player_sum_grid, dealer_showing_grid, value_function, cmap="viridis"
        )

        # Set labels and title
        ax.set_xlabel("Player Sum")
        ax.set_ylabel("Dealer Showing")
        ax.set_zlabel("Value Function")
        ax.set_title("3D Surface Plot of Value Function in Easy21")

        plt.show()


def main():
    mc_control = MCControl()
    mc_control.train(1000000)


if __name__ == "__main__":
    main()
