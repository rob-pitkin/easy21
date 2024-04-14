# Author: Rob Pitkin
# Date: 4/14/2024
# Description: This file contains the main game loop and logic for the game.

from typing import List
import random
import time

class Card:
    def __init__(self, value: str, color: str):
        self.value = value
        self.color = color
    
    def __repr__(self) -> str:
        return f"{self.color} {self.value}"

class GameState:
    def __init__(self, dealer_card: Card, player_sum: int):
        self.dealer_card = dealer_card
        self.player_sum = player_sum
   
    def __repr__(self) -> str:
        return f"Dealer: {self.dealer_card}, Player: {self.player_sum}"

class Easy21:
    def __init__(self):
        random.seed(time.time())
        self.dealer_cards: List[Card] = [self.draw_first_card()]
        self.player_cards: List[Card] = [self.draw_first_card()]

    def draw_card(self) -> Card:
        # Draw a number from the deck (1-10) and assign it a color
        num = random.randint(1, 10)
        color = "red" if random.random() < 1/3 else "black"
        return Card(str(num), color)
    
    def draw_first_card(self) -> Card:
        # Draw the first card for the player or dealer
        num = random.randint(1, 10)
        color = "black"
        return Card(str(num), color)
    
    def get_sum(self, cards: List[Card]) -> int:
        # Calculate the sum of the cards
        total = 0
        for card in cards:
            if card.color == "red":
                total -= int(card.value)
            else:
                total += int(card.value)
        return total
    
    def decide_winner(self) -> int:
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
    
    def step(self, state: GameState, action: str) -> tuple[GameState, int]:
        print(f"Current state: {state}")
        dealer_card, player_sum = state.dealer_card, state.player_sum
        # If the player sticks, the dealer draws until their sum is greater than 17
        if action == "stick":
            while self.get_sum(self.dealer_cards) < 17:
                self.dealer_cards.append(self.draw_card())
            return (dealer_card, player_sum), self.decide_winner()
        # If the player hits, the player draws a card
        elif action == "hit":
            self.player_cards.append(self.draw_card())
            player_sum = self.get_sum(self.player_cards)
            # If the player's sum is greater than 21, the player loses
            if player_sum > 21:
                return (dealer_card, player_sum), -1
            return (dealer_card, player_sum), 0
