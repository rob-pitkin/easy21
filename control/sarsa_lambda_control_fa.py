# Author: Rob Pitkin
# Date: 05/12/2024
# Description: This file contains the Sarsa Lambda control logic for the game using function approximation.

import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from monte_carlo_control import MCControl
from game import Card, Easy21, GameState
import torch
import torch.nn as nn
import numpy as np
from utils import MSEParams


class SarsaLambdaLinearNetwork(nn.Module):
    """
    A class to represent the linear network for function approximation

    Attributes:
        input_size (int): The size of the input
        output_size (int): The size of the output

    Methods:
        forward: Perform a forward pass through the network
    """

    def __init__(self, input_size: int, output_size: int):
        super(SarsaLambdaLinearNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SarsaLambdaControlFA:
    """
    A class to represent the Sarsa Lambda control agent using function approximation

    Attributes:
        function_approximator (SarsaLambdaLinearNetwork): The function approximator
        gamma (float): The discount factor
        lambda_ (float): The lambda value
        lr (float): The learning rate
        eligibility_traces (np.array): The eligibility traces
        state_action_space (List[tuple]): The state-action space
        epsilon (float): The epsilon value

    Methods:
        init_eleigibility_traces: Initialize the eligibility traces
        get_input: Get the input for the function approximator
        update_weights: Update the weights of the function approximator
        get_action: Get the action for the given state
        train: Train the agent using Sarsa(lambda)
        calculate_mean_squared_error: Calculate the mean squared error between the Sarsa and Monte Carlo Q-values
        plot_q_values_mse_against_episodes: Plot the MSE of the Q-values against the num of episodes
        plot_q_values_mse_against_lambda: Plot the MSE of the Q-values against the lambda value

    """

    def __init__(
        self,
        function_approximator: SarsaLambdaLinearNetwork,
        lambda_: float,
        gamma=1.0,
        lr=0.01,
        epsilon=0.05,
    ) -> None:
        """
        Initialize the Sarsa Lambda control agent

        Args:
            function_approximator (SarsaLambdaLinearNetwork): The function approximator
            lambda_ (float): The lambda value
            gamma (float, optional): The discount factor. Defaults to 1.0.
            lr (float, optional): The learning rate. Defaults to 0.01.
            epsilon (float, optional): The epsilon value. Defaults to 0.05.

        Returns:
            None
        """
        self.function_approximator = function_approximator
        self.gamma = gamma
        self.lambda_ = lambda_
        self.lr = lr
        self.eligibility_traces = np.zeros(
            self.function_approximator.linear.weight.shape[1], dtype=np.float32
        )
        dealer_intervals = [(1, 4), (4, 7), (7, 10)]
        player_sum_intervals = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
        actions = ["hit", "stick"]
        self.state_action_space = [
            (dealer_interval, player_sum_interval, action)
            for dealer_interval in dealer_intervals
            for player_sum_interval in player_sum_intervals
            for action in actions
        ]
        self.epsilon = epsilon

    def init_eleigibility_traces(self) -> None:
        """
        Initialize the eligibility traces

        Returns:
            None
        """
        self.eligibility_traces = np.zeros(
            self.function_approximator.linear.weight.shape[1], dtype=np.float32
        )

    def get_input(self, state: GameState, action: str) -> torch.Tensor:
        """
        Get the input for the function approximator

        Args:
            state (GameState): The state
            action (str): The action

        Returns:
            torch.Tensor: The input for the function approximator
        """
        feature_vector = [0] * self.function_approximator.linear.weight.shape[1]
        dealer_card = state.dealer_card.value
        player_sum = state.player_sum
        for i, (d, p, a) in enumerate(self.state_action_space):
            if (
                d[0] <= dealer_card <= d[1]
                and (p[0] <= player_sum <= p[1])
                and a == action
            ):
                feature_vector[i] = 1
        return torch.tensor(feature_vector, dtype=torch.float32)

    def update_weights(
        self,
        state: GameState,
        action: str,
        reward: int,
        next_state: GameState,
        next_action: str,
    ) -> None:
        """
        Update the weights of the function approximator

        Args:
            state (GameState): The current state
            action (str): The action
            reward (int): The reward
            next_state (GameState): The next state
            next_action (str): The next action

        Returns:
            None
        """
        if reward == -1:
            next_state_value = 0.0
        else:
            with torch.no_grad():
                next_state_value = self.function_approximator(
                    self.get_input(next_state, next_action)
                ).item()
        with torch.no_grad():
            delta = (
                reward
                + self.gamma * next_state_value
                - self.function_approximator(self.get_input(state, action)).item()
            )
            self.eligibility_traces = (
                self.gamma * (self.lambda_ * self.eligibility_traces)
                + self.get_input(state, action).numpy()
            )
            self.function_approximator.linear.weight += torch.tensor(
                self.lr * (self.gamma * delta) * self.eligibility_traces
            )

    def get_action(self, state: GameState) -> str:
        """
        Get the e-greedy action for the given state

        Args:
            state (GameState): The state

        Returns:
            str: The action
        """
        if random.random() < self.epsilon:
            return random.choice(["hit", "stick"])
        with torch.no_grad():
            hit_value = self.function_approximator(self.get_input(state, "hit")).item()
            stick_value = self.function_approximator(
                self.get_input(state, "stick")
            ).item()
        return "hit" if hit_value > stick_value else "stick"

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
            if (i + 1) % 100 == 0:
                print(
                    f"Training episode {i + 1}/{num_episodes} with lambda {self.lambda_:.1f}"
                )
            self.init_eleigibility_traces()
            game = Easy21()
            state, action = game.get_state(), self.get_action(game.get_state())
            while not game.is_finished:
                next_state, reward = game.step(state, action or "")
                next_action = self.get_action(next_state)
                self.update_weights(state, action, reward, next_state, next_action)
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
        player_sums = range(1, 22)
        dealer_cards = range(1, 11)
        actions = ["hit", "stick"]
        for player_sum in player_sums:
            for dealer_card in dealer_cards:
                for action in actions:
                    state = GameState(Card(dealer_card, "black"), player_sum)
                    with torch.no_grad():
                        sarsa_q_value = self.function_approximator(
                            self.get_input(state, action)
                        ).item()
                    mc_q_value = MC_q_values.get((state, action), 0)
                    mse += (sarsa_q_value - mc_q_value) ** 2
        return mse / len(MC_q_values)

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
        plt.title(f"Sarsa({self.lambda_}) Mean Squared Error")
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
    lambda_vals = [0.0, 1.0]
    for lambda_val in lambda_vals:
        mse_values = []
        function_approximator = SarsaLambdaLinearNetwork(36, 1)
        sarsa_lambda_control = SarsaLambdaControlFA(function_approximator, lambda_val)
        mse_params = MSEParams(mse_values, MC_control_q_values)
        sarsa_lambda_control.train(1000, mse_params)
        sarsa_lambda_control.plot_q_values_mse_against_episodes(mse_values)


def main():
    """
    Main function to train the Sarsa Lambda control agent and show MSE plots
    """
    mc_control = MCControl()
    mc_control.train(1000000)
    train_and_show_plots(mc_control.q_values)


if __name__ == "__main__":
    main()
