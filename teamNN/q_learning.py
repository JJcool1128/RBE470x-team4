import numpy as np
import random

class ApproximateQLearningAgent:
    def __init__(self, actions, feature_extractor, alpha=0.01, gamma=0.9, epsilon=0.1):
        """
        :param actions: List of possible actions
        :param feature_extractor: Function that extracts features from (state, action) pairs
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration probability
        """
        self.actions = actions
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights = {}  # Dictionary to store feature weights

    def get_q_value(self, state, action):
        """Compute Q(s, a) using linear approximation."""
        features = self.feature_extractor(state, action)
        return sum(self.weights.get(f, 0) * v for f, v in features.items())

    def update(self, state, action, reward, next_state):
        """Perform the Q-learning update step."""
        features = self.feature_extractor(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in self.actions)

        # Compute TD Error
        delta = (reward + self.gamma * max_next_q) - self.get_q_value(state, action)

        # Update weights
        for f, value in features.items():
            if f not in self.weights:
                self.weights[f] = 0
            self.weights[f] += self.alpha * delta * value

    def select_action(self, state):
        """Select an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.get_q_value(state, a))

# Example Feature Extractor
def simple_feature_extractor(state, action):
    features = {
        "bias": 1.0,  # Bias term for weight adjustment
        "dist_to_monster": 1 / (state.dist_to_closest_ghost() + 1),
        "dist_to_exit": 1 / (state.dist_to_closest_food() + 1),
        "dist_to_bomb": 1 / (state.dist_to_closest_bomb() + 1),
        f"action_{action}": 1  # Encodes the action in features
    }
    return features

# Usage Example
actions = ["UP", "DOWN", "LEFT", "RIGHT"]
agent = ApproximateQLearningAgent(actions, simple_feature_extractor)

state = ... 
action = agent.select_action(state)
reward = ...  
next_state = ... 
agent.update(state, action, reward, next_state)
