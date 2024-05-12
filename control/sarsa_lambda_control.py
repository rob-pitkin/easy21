# Author: Rob Pitkin
# Date: 05/12/2024
# Description: This file contains the Sarsa Lambda control logic for the game.

import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from game import Card, Easy21, GameState
from monte_carlo_control import MCControl


class MSEParams:
    """
    A class to represent the parameters for calculating the mean squared error

    Attributes:
        mse_values (List[float]): The mean squared error values
        q_values (Dict[tuple[GameState, str], float]): The state-action value function
    """

    def __init__(
        self, mse_values: List[float], q_values: Dict[Tuple[GameState, str], float]
    ) -> None:
        self.mse_values = mse_values
        self.q_values = q_values


class SarsaLambdaControl:
    """
    A class to represent the Sarsa Lambda control agent

    Attributes:
        eligibility_traces (Dict[tuple[GameState, str], float]): The eligibility traces
        n0 (float): Epsilon constant for epsilon-greedy policy
        lambda_val (float): The lambda value for Sarsa(lambda)
        q_values (Dict[tuple[GameState, str], float]): The state-action value function
        n_state (Dict[GameState, int]): The number of times a state has been visited
        n_state_action (Dict[tuple[GameState, str], int]): The number of times a state-action
            pair has been visited

    Methods:
        get_epsilon: Get the epsilon value for the given state
        init_eleigibility_traces: Initialize the eligibility traces to 0
        get_alpha: Get the alpha value for the given state-action pair
        get_q_value: Get the Q-value for the given state-action pair
        get_epsilon_greedy_action: Get the best action for the given state using an
            epsilon-greedy policy
        train: Train the agent using Sarsa(lambda)
        calculate_mean_squared_error: Calculate the mean squared error between the
            Sarsa and Monte Carlo Q-values
        plot_q_values_mse_against_episodes: Plot the MSE of the Q-values against the num of episodes
        plot_q_values_mse_against_lambda: Plot the MSE of the Q-values against the lambda value
    """

    def __init__(self, lambda_val) -> None:
        self.eligibility_traces: Dict[Tuple[GameState, str], float] = {}
        self.n0: float = 100
        self.lambda_val: float = lambda_val

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

    def get_epsilon(self, state: GameState) -> float:
        """
        Get the epsilon value for the given state

        Args:
            state (GameState): The state to get the epsilon value for

        Returns:
            float: The epsilon value for the given state
        """
        return self.n0 / (self.n0 + self.n_state[state])

    def init_eleigibility_traces(self) -> None:
        """
        Initialize the eligibility traces to 0
        """
        for dealer_card in range(1, 11):
            for player_sum in range(1, 22):
                for action in ["hit", "stick"]:
                    self.eligibility_traces[
                        (GameState(Card(dealer_card, "black"), player_sum), action)
                    ] = 0

    def get_alpha(self, state: GameState, action: str) -> float:
        """
        Get the alpha value for the given state-action pair

        Args:
            state (GameState): The state to get the alpha value for
            action (str): The action to get the alpha value for

        Returns:
            float: The alpha value for the given state-action pair
        """
        return 1 / (1 + self.n_state_action[(state, action)])

    def get_q_value(self, state: GameState, action: str) -> float:
        """
        Get the Q-value for the given state-action pair

        Args:
            state (GameState): The state to get the Q-value for
            action (str): The action to get the Q-value for

        Returns:
            float: The Q-value for the given state-action pair
        """
        return self.q_values[(state, action)]

    def get_epsilon_greedy_action(self, state: GameState) -> str:
        """
        Get the best action for the given state using an epsilon-greedy policy

        Args:
            state (GameState): The state to get the best action for

        Returns:
            str: The best action for the given state
        """
        epsilon = self.get_epsilon(state)
        max_q = float("-inf")
        best_action = ""
        if random.random() < epsilon:
            best_action = random.choice(["hit", "stick"])
        else:
            for action in ["hit", "stick"]:
                q = self.get_q_value(state, action)
                if q > max_q:
                    max_q = q
                    best_action = action
        return best_action

    def train(
        self,
        num_episodes: int,
        mse_params: MSEParams = None,  # type: ignore
    ) -> None:
        """
        Train the agent using Sarsa(lambda)

        Args:
            num_episodes (int): The number of episodes to train for
            mse_params (MSEParams, optional): The parameters to calculate the
                mean squared error. Defaults to None.
        """
        for i in range(num_episodes):
            if (i + 1) % 1000 == 0:
                print(
                    f"Training episode {i + 1}/{num_episodes} with lambda {self.lambda_val:.1f}"
                )
            self.init_eleigibility_traces()
            game = Easy21()
            state, action = game.get_state(), self.get_epsilon_greedy_action(
                game.get_state()
            )
            while not game.is_finished:
                self.n_state[state] += 1
                self.n_state_action[(state, action or "")] += 1
                next_state, reward = game.step(state, action or "")
                next_action = None
                if game.is_finished:
                    delta = reward - self.get_q_value(state, action or "")
                else:
                    next_action = self.get_epsilon_greedy_action(next_state)
                    delta = (
                        reward
                        + self.get_q_value(next_state, next_action)
                        - self.get_q_value(state, action or "")
                    )
                self.eligibility_traces[(state, action or "")] += 1
                for dealer_card in range(1, 11):
                    for player_sum in range(1, 22):
                        for a in ["hit", "stick"]:
                            state = GameState(Card(dealer_card, "black"), player_sum)
                            self.q_values[(state, a)] += (
                                self.get_alpha(state, a)
                                * delta
                                * self.eligibility_traces[(state, a)]
                            )
                            self.eligibility_traces[(state, a)] *= self.lambda_val
                state, action = next_state, next_action
            if mse_params:
                mse = self.calculate_mean_squared_error(mse_params.q_values)
                mse_params.mse_values.append(mse)

    def calculate_mean_squared_error(
        self, MC_q_values: Dict[Tuple[GameState, str], float]
    ) -> float:
        """
        Calculate the mean squared error between the Sarsa and Monte Carlo Q-values

        Args:
            MC_q_values (Dict): The Monte Carlo Q-values

        Returns:
            float: The mean squared error between the Sarsa and Monte Carlo Q-values
        """
        mse = 0
        for state, value in self.q_values.items():
            mse += (value - MC_q_values[state]) ** 2
        return mse / len(self.q_values)

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
        plt.title(f"Sarsa({self.lambda_val}) Mean Squared Error")
        plt.show()

    def plot_q_values_mse_against_lambda(
        self, mse_values_per_lambda: List[float], lambda_vals: List[float]
    ) -> None:
        """
        Plot the MSE of the Q-values against the lambda value

        Args:
            mse_values_per_lambda (List[float]): The MSE values per lambda
            lambda_vals (List[float]): The lambda values

        Returns:
            None
        """
        plt.plot(lambda_vals, mse_values_per_lambda)
        plt.xlabel("Lambda Value")
        plt.ylabel("Mean Squared Error")
        plt.title("Sarsa Lambda Mean Squared Error")
        plt.show()


def train_and_show_plots(MC_control_q_values: Dict[Tuple[GameState, str], float]):
    """
    Train the Sarsa Lambda control agent and show the plots

    Args:
        MC_control_q_values (Dict): The Monte Carlo control Q-values

    Returns:
        None
    """
    # Store lambdas from 0 to 1 in increments of 0.1
    lambda_vals = np.linspace(0, 1, 11)
    mse_values_per_lambda = []
    for lambda_val in lambda_vals:
        mse_values = []
        sarsa_lambda_control = SarsaLambdaControl(lambda_val)
        mse_params = MSEParams(mse_values, MC_control_q_values)
        sarsa_lambda_control.train(1000, mse_params)
        if lambda_val == 0 or lambda_val == 1:
            sarsa_lambda_control.plot_q_values_mse_against_episodes(mse_values)
        mse_values_per_lambda.append(mse_values[-1])
    sarsa_lambda_control.plot_q_values_mse_against_lambda(
        mse_values_per_lambda, lambda_vals.tolist()
    )


def main():
    """
    Main function to train the Sarsa Lambda control agent and show MSE plots
    """
    mc_control = MCControl()
    mc_control.train(100000)
    train_and_show_plots(mc_control.q_values)


if __name__ == "__main__":
    main()
