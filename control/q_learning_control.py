# Author: Rob Pitkin
# Date: 05/12/2024
# Description: This file contains the Q-learning control logic for the game.

import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from game import Card, Easy21, GameState
from monte_carlo_control import MCControl
from utils import MSEParams


class QLearningControl:
    """
    A class to represent the Q-learning control agent

    Attributes:
        n0 (float): Epsilon constant for epsilon-greedy policy
        q_values (Dict[tuple[GameState, str], float]): The state-action value function
        n_state (Dict[GameState, int]): The number of times a state has been visited
        n_state_action (Dict[tuple[GameState, str], int]): The number of times a state-action
            pair has been visited

    Methods:
        get_epsilon: Get the epsilon value for the given state
        get_alpha: Get the alpha value for the given state-action pair
        get_q_value: Get the Q-value for the given state-action pair
        get_epsilon_greedy_action: Get the best action for the given state using an
            epsilon-greedy policy
        train: Train the agent using Q-learning
        calculate_mean_squared_error: Calculate the mean squared error between the
            Q-learning and Monte Carlo Q-values
        plot_q_values_mse_against_episodes: Plot the MSE of the Q-values against the num of episodes
    """

    def __init__(self) -> None:
        self.n0: float = 100

        self.q_values: Dict[Tuple[GameState, str], float] = {}
        for dealer_card in range(1, 11):
            for player_sum in range(1, 22):
                for action in ["hit", "stick"]:
                    self.q_values[
                        (GameState(Card(dealer_card, "black"), player_sum), action)
                    ] = 0

        self.n_state: Dict[GameState, int] = {}
        for dealer_card in range(1, 11):
            for player_sum in range(1, 22):
                self.n_state[GameState(Card(dealer_card, "black"), player_sum)] = 0

        self.n_state_action: Dict[Tuple[GameState, str], int] = {}
        for dealer_card in range(1, 11):
            for player_sum in range(1, 22):
                for action in ["hit", "stick"]:
                    self.n_state_action[
                        (GameState(Card(dealer_card, "black"), player_sum), action)
                    ] = 0

    def get_alpha(self, state_action: Tuple[GameState, str]) -> float:
        """
        Get the alpha value for the given state-action pair

        Args:
            state_action (tuple[GameState, str]): The state-action pair

        Returns:
            float: The alpha value
        """
        return 1 / (1 + self.n_state_action[state_action])

    def get_q_value(self, state_action: Tuple[GameState, str]) -> float:
        """
        Get the Q-value for the given state-action pair

        Args:
            state_action (tuple[GameState, str]): The state-action pair

        Returns:
            float: The Q-value
        """
        return self.q_values[state_action]

    def get_episilon(self, state: GameState) -> float:
        """
        Get the epsilon value for the given state

        Args:
            state (GameState): The state to get the epsilon value for

        Returns:
            float: The epsilon value for the given state
        """
        return self.n0 / (self.n0 + self.n_state[state])

    def get_e_greedy_action(self, state: GameState) -> str:
        """
        Get the best action for the given state using an epsilon-greedy policy

        Args:
            state (GameState): The state to get the action for

        Returns:
            str: The best action
        """
        epsilon = self.get_episilon(state)
        if random.random() < epsilon:
            return random.choice(["hit", "stick"])
        else:
            if self.get_q_value((state, "hit")) > self.get_q_value((state, "stick")):
                return "hit"
            else:
                return "stick"

    def calculate_mean_squared_error(
        self, mc_q_values: Dict[Tuple[GameState, str], float]
    ) -> float:
        """
        Calculate the mean squared error between the Q-learning and Monte Carlo Q-values

        Args:
            mc_q_values (Dict[tuple[GameState, str], float]): The Monte Carlo Q-values

        Returns:
            float: The mean squared error
        """
        mse = 0
        for state_action in self.q_values:
            mse += (self.q_values[state_action] - mc_q_values[state_action]) ** 2
        return mse / len(self.q_values)

    def train(
        self,
        num_episodes: int,
        mse_params: MSEParams = None,  # type: ignore
    ) -> None:
        """
        Train the agent using Q-learning

        Args:
            num_episodes (int): The number of episodes to train for
            mse_params (MSEParams): The parameters for calculating the mean squared error

        Returns:
            None
        """
        for i in range(num_episodes):
            if (i + 1) % 100 == 0:
                print(f"Training episode {i + 1}/{num_episodes}")
            game = Easy21()
            state = game.get_state()
            while not game.is_finished:
                self.n_state[state] += 1
                action = self.get_e_greedy_action(state)
                self.n_state_action[(state, action or "")] += 1
                next_state, reward = game.step(state, action)
                if not game.is_finished:
                    self.q_values[(state, action)] += self.get_alpha(
                        (state, action)
                    ) * (
                        reward
                        + max(
                            self.get_q_value((next_state, "hit")),
                            self.get_q_value((next_state, "stick")),
                        )
                        - self.get_q_value((state, action))
                    )
                    state = next_state
                else:
                    self.q_values[(state, action)] += self.get_alpha(
                        (state, action)
                    ) * (reward - self.get_q_value((state, action)))
            if mse_params is not None:
                mse_params.mse_values.append(
                    self.calculate_mean_squared_error(mse_params.q_values)
                )

    def plot_q_values_mse_against_episodes(self, mse_values: List[float]) -> None:
        """
        Plot the MSE of the Q-values against the num of episodes

        Args:
            mse_values (List[float]): The MSE values

        Returns:
            None
        """
        plt.plot(range(1, len(mse_values) + 1), mse_values)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Q-learning Mean Squared Error")
        plt.show()


def main():
    num_episodes = 1000
    q_learning_control = QLearningControl()
    mc_control = MCControl()
    mc_control.train(1000000)
    mse_params = MSEParams([], mc_control.q_values)
    q_learning_control.train(num_episodes, mse_params)
    q_learning_control.plot_q_values_mse_against_episodes(mse_params.mse_values)
    # q_learning_control.train(num_episodes)


if __name__ == "__main__":
    main()
