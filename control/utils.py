from typing import Dict, List, Tuple
from game import GameState


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
