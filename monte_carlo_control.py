# Author: Rob Pitkin
# Date: 4/14/2024
# Description: This file contains the Monte Carlo control logic for the game.

from game import Easy21, GameState
from typing import Dict, List
import random


class MCControl:
    def __init__(self, n_0=100):
        self.q_values: Dict[tuple[GameState, str], float] = (
            {}
        )  # State-action value function
        self.n_state: Dict[GameState, int] = (
            {}
        )  # Number of times a state has been visited
        self.n_state_action: Dict[tuple[GameState, str], int] = (
            {}
        )  # Number of times a state-action pair has been visited
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
        return self.n_0 / (
            self.n_0 + (self.n_state[state] if state in self.n_state else 0)
        )

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
        return (
            1 / self.n_state_action[(state, action)]
            if (state, action) in self.n_state_action
            else 1
        )

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
        return self.q_values[(state, action)] if (state, action) in self.q_values else 0

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
        while not game.is_finished:
            state = game.get_state()
            action = self.get_e_greedy_action(state)
            _, reward = game.step(state, action)
            episode.append((state, action, reward))
        return episode

    def update_q_values(self, episode: List[tuple[GameState, str, int]]) -> bool:
        """
        Update the Q-values for the episode

        Args:
            episode (List[tuple[GameState, str, int]]): The episode to update the Q-values for

        Returns:
            bool: True if the Q-values have converged, False otherwise
        """
        # Update the Q-values for the episode
        G_t = 0
        max_diff = 0
        # Iterate through the episode in reverse
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            G_t += reward
            self.n_state[state] = (
                self.n_state[state] + 1 if state in self.n_state else 1
            )
            self.n_state_action[(state, action)] = (
                self.n_state_action[(state, action)] + 1
                if (state, action) in self.n_state_action
                else 1
            )
            alpha = self.get_alpha(state, action)
            q = self.get_q_value(state, action)
            self.q_values[(state, action)] = q + alpha * (G_t - q)
            max_diff = max(max_diff, abs(q - self.q_values[(state, action)]))
        return max_diff < 1e-6

    def train(self, n_episodes=1000) -> None:
        """
        Train the agent using Monte Carlo control

        Args:
            n_episodes (int): The number of episodes to train the agent for

        Returns:
            None
        """
        converged = False
        # Train the agent
        for i in range(n_episodes):
            print(f"Training episode {i + 1}/{n_episodes}")
            episode = self.sample_episode()
            self.update_q_values(episode):


def main():
    mc_control = MCControl()
    mc_control.train(100000)


if __name__ == "__main__":
    main()
