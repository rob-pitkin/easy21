# Author: Rob Pitkin
# Date: 4/14/2024
# Description: This file contains the main game loop and logic for the game.

from typing import List
import random
import time
from typing import Tuple


class Card:
    """
    A class to represent a card in the game

    Attributes:
        value (int): The value of the card
        color (str): The color of the card

    Methods:
        __repr__: Return the string representation of the card
        __hash__: Return the hash value of the card
        __eq__: Check if the card is equal to another card
    """

    def __init__(self, value: int, color: str):
        self.value = value
        self.color = color

    def __repr__(self) -> str:
        return f"{self.color} {self.value}"

    def __hash__(self) -> int:
        if self.color == "red":
            return hash(-self.value)
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.value == other.value and self.color == other.color


class GameState:
    """
    A class to represent the state of the game

    Attributes:
        dealer_card (Card): The dealer's card
        player_sum (int): The sum of the player's cards

    Methods:
        __repr__: Return the string representation of the game state
        __hash__: Return the hash value of the game state
        __eq__: Check if the game state is equal to another game state
    """

    def __init__(self, dealer_card: Card, player_sum: int):
        self.dealer_card = dealer_card
        self.player_sum = player_sum

    def __repr__(self) -> str:
        return f"Dealer: {self.dealer_card}, Player: {self.player_sum}"

    def __hash__(self) -> int:
        return hash((self.dealer_card, self.player_sum))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GameState):
            return False
        return (
            self.dealer_card == other.dealer_card
            and self.player_sum == other.player_sum
        )


class Easy21:
    def __init__(self):
        random.seed(time.time())
        self.dealer_cards: List[Card] = [self.draw_first_card()]
        self.player_cards: List[Card] = [self.draw_first_card()]
        self.is_finished = False

    def draw_card(self) -> Card:
        """
        Draw a card from the deck

        Returns:
            Card: The card drawn from the deck
        """
        # Draw a number from the deck (1-10) and assign it a color
        num = random.randint(1, 10)
        color = "red" if random.random() < 1 / 3 else "black"
        return Card(num, color)

    def draw_first_card(self) -> Card:
        """
        Draw the first card for the player or dealer

        Returns:
            Card: The first card drawn for the player or dealer
        """
        # Draw the first card for the player or dealer
        num = random.randint(1, 10)
        color = "black"
        return Card(num, color)

    def get_state(self) -> GameState:
        """
        Get the current state of the game

        Returns:
            GameState: The current state of the game
        """
        # Get the current state of the game
        return GameState(self.dealer_cards[0], self.get_sum(self.player_cards))

    def get_sum(self, cards: List[Card]) -> int:
        """
        Get the sum of the cards

        Args:
            cards (List[Card]): The cards to calculate the sum for

        Returns:
            int: The sum of the cards
        """
        # Calculate the sum of the cards
        total = 0
        for card in cards:
            if card.color == "red":
                total -= int(card.value)
            else:
                total += int(card.value)
        return total

    def decide_winner(self) -> int:
        """
        Decide the winner of the game

        Returns:
            int: 1 if the player wins, -1 if the player loses, 0 if it's a draw
        """
        self.is_finished = True
        # If the player's sum is greater than 21, the player loses
        player_sum = self.get_sum(self.player_cards)
        if player_sum > 21:
            return -1
        # If the dealer's sum is greater than 21, the player wins
        dealer_sum = self.get_sum(self.dealer_cards)
        if dealer_sum > 21:
            return 1
        # If the player's sum is greater than the dealer's sum, the player wins
        if player_sum > dealer_sum:
            return 1
        # If the player's sum is less than the dealer's sum, the player loses
        if player_sum < dealer_sum:
            return -1
        # If the player's sum is equal to the dealer's sum, it's a draw
        return 0

    def step(self, state: GameState, action: str) -> Tuple[GameState, int]:
        """
        Take a step in the game

        Args:
            state (GameState): The current state of the game
            action (str): The action to take in the game

        Returns:
            tuple[GameState, int]: The new state of the game and the reward for the action taken
        """
        dealer_card, player_sum = state.dealer_card, state.player_sum
        # If the player sticks, the dealer draws until their sum is greater than 17
        if action == "stick":
            while self.get_sum(self.dealer_cards) < 17:
                self.dealer_cards.append(self.draw_card())
            return GameState(dealer_card, player_sum), self.decide_winner()
        # If the player hits, the player draws a card
        elif action == "hit":
            self.player_cards.append(self.draw_card())
            player_sum = self.get_sum(self.player_cards)
            # If the player's sum is greater than 21, the player loses
            if player_sum > 21 or player_sum < 1:
                self.is_finished = True
                return GameState(dealer_card, player_sum), -1
            return GameState(dealer_card, player_sum), 0
        else:
            raise ValueError("Invalid action")
