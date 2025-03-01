import sys
sys.path.insert(0, '../bomberman')
from entity import CharacterEntity
import random
import numpy as np
from collections import deque

# Import PyTorch for the neural network implementation.
import torch
import torch.nn as nn
import torch.optim as optim

# Define the DQN neural network. This network takes a state vector (here of length 8)
# and outputs Q-values for 6 discrete actions:
#   0: move up, 1: move down, 2: move left, 3: move right, 4: stay, 5: place bomb.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q-Learning based Bomberman character.
class TestCharacter(CharacterEntity):
    print("DQN TestCharacter class loaded!")
    
    # --- DQN Hyperparameters and shared objects ---
    input_dim = 8    # Size of our state vector (see get_state below)
    output_dim = 6   # Number of actions (up, down, left, right, stay, bomb)
    # Our Q-network is defined as a class variable so that its memory and training
    # persist across episodes.
    # Our main Q-network
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=2000)

    # Add Target Network (to stabilize learning)
    target_model = DQN(input_dim, output_dim)
    target_model.load_state_dict(model.state_dict())  # Copy initial weights
    target_model.eval()  # Target network is only used for computing Q-targets

    # DQN Hyperparameters
    gamma = 0.99
    epsilon = 1.0       # Initial exploration rate
    epsilon_min = 0.1
    epsilon_decay = 0.995  # Slower decay to prevent early exploitation
    batch_size = 32
    target_update_freq = 50  # Update target network every 50 episodes

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        # To record the previous state and action for the transition update.
        self.last_state = None
        self.last_action = None
        try:
            self.__class__.target_model.load_state_dict(self.__class__.model.state_dict())
            # self.__class__.model.eval()  # Switch to evaluation mode
            # self.__class__.epsilon = 0.05  # Reduce randomness for deployment
            print("Loaded trained model successfully!")
        except FileNotFoundError:
            print("No pre-trained model found. Training from scratch.")

    def neighbors_of_4(self, wrld, current: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = []
        a, b = current
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = a + dx, b + dy
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                if not wrld.wall_at(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    # --- State representation ---
    def get_state(self, wrld):
        """
        Build a simple state vector:
         - Agent's x and y normalized by grid width/height.
         - Exit cell's x and y normalized.
         - Relative x and y difference (normalized) and Manhattan distance (normalized)
           to the closest monster (if any).
         - 1.0 if a bomb is on the current cell, else 0.0.
        """
        width = wrld.width()
        height = wrld.height()
        agent_x = self.x / width
        agent_y = self.y / height
        exit_x = wrld.exitcell[0] / width
        exit_y = wrld.exitcell[1] / height

        # Find the closest monster, if any.
        closest_monster = None
        min_dist = float('inf')
        for monster_list in wrld.monsters.values():
            for m in monster_list:
                dist = abs(self.x - m.x) + abs(self.y - m.y)
                if dist < min_dist:
                    min_dist = dist
                    closest_monster = m
        if closest_monster is None:
            monster_dx = 0.0
            monster_dy = 0.0
            monster_dist = 1.0  # default value when no monster is nearby
        else:
            monster_dx = (closest_monster.x - self.x) / width
            monster_dy = (closest_monster.y - self.y) / height
            monster_dist = min_dist / (width + height)
        bomb_here = 1.0 if wrld.bomb_at(self.x, self.y) else 0.0

        state = np.array([agent_x, agent_y, exit_x, exit_y,
                          monster_dx, monster_dy, monster_dist, bomb_here],
                         dtype=np.float32)
        return state

    # --- Reward function ---
    def get_reward(self, wrld, state, next_state):
        """
        A basic reward function:
        - Small penalty each move (-0.1).
        - Bonus for reaching the exit (+10).
        - Penalty for being caught in an explosion (-10).
        - Additional penalty if a monster is very close.
        - One-time bonus for crossing a section of walls.
        """
        reward = 0
        done = False

        # Reward for reaching the exit.
        if (self.x, self.y) == wrld.exitcell:
            reward += 10.0
            done = True

        # Penalty if the agent is hit by an explosion.
        if wrld.explosion_at(self.x, self.y):
            reward -= 10.0
            done = True

        # for dx in [-1,1]:
        #     if wrld.bomb_at(self.x + dx, self.y) and not wrld.bomb_at(self.x, self.y):
        #         reward -= 10.0
        #         break
            
        # for dy in [-1,1]:
        #     if wrld.bomb_at(self.x, self.y + dy) and not wrld.bomb_at(self.x, self.y):
        #         reward -= 10.0
        #         break

        # Extra penalty if a monster is dangerously close.
        for monster_list in wrld.monsters.values():
            for m in monster_list:
                if abs(self.x - m.x) + abs(self.y - m.y) < 2:
                    reward -= 2.0
                if self.x == m.x and self.y == m.y:
                    reward -= 10.0
                    done = True

        # --- Additional one-time reward for crossing a section of walls ---
        # Assume wall sections are at rows 3, 7, 11, and 15.
        # Convert normalized y-coordinates back to grid rows.

        # Distance-based reward: Encourage movement towards the exit.
        goal = (wrld.exitcell[0], wrld.exitcell[1])
        
        # Manhattan distance to the goal
        current_distance = abs(self.x - goal[0]) + abs(self.y - goal[1])
        prev_distance = abs(self.last_state[0] * wrld.width() - goal[0]) + abs(self.last_state[1] * wrld.height() - goal[1]) if self.last_state is not None else current_distance

        # Reward for getting closer to the goal.
        if current_distance < prev_distance:
            reward += 0.5 # Increase this if the agent needs stronger guidance.

        # Penalty if moving away from the goal.
        elif current_distance > prev_distance:
            reward -= 0.5  # Increase this if the agent frequently backtracks.

        # Check if there is a monster within a Euclidean distance of less than 4.
        monster_nearby = False
        for monster_list in wrld.monsters.values():
            for m in monster_list:
                dist = ((self.x - m.x)**2 + (self.y - m.y)**2)**0.5
                if dist < 4:
                    monster_nearby = True
                    break
            if monster_nearby:
                break
        
        # wall_corner = False
        
        # if wrld.wall_at(self.x,self.y+1) and (wrld.width() - self.x == 1):
        #     wall_corner = True

        # Place bomb if a monster is close or if there's no available path.
        # if monster_nearby:
        #     if (self.wall_blocked(wrld)):
        #         if state[7] == 1.0:
        #             reward += 2

        if state[7] == 1.0:
            reward -= 1
        #if (wrld.bomb_at(self.x, self.y) and not wall_corner and not monster_nearby):
        #    reward -= 50

        return reward, done

    # --- Experience replay helper ---
    def remember(self, state, action, reward, next_state, done):
        self.__class__.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        # if len(self.__class__.memory) < self.__class__.batch_size:
        #     return  # Not enough samples to train.
        if len(self.__class__.memory) >= self.__class__.batch_size:
            batch = random.sample(self.__class__.memory, self.__class__.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(np.array(states), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.bool)

            # Compute Q-values and targets
            q_values = self.__class__.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.__class__.target_model(next_states).max(1)[0]
            target = rewards + self.__class__.gamma * next_q_values * (~dones)

            # Compute loss and optimize
            loss = self.__class__.loss_fn(q_values, target.detach())
            self.__class__.optimizer.zero_grad()
            loss.backward()
            self.__class__.optimizer.step()

            # Decay epsilon
            # Only decay epsilon once per episode, not per step
            if len(self.__class__.memory) >= self.__class__.batch_size:
                if self.__class__.epsilon > self.__class__.epsilon_min:
                    self.__class__.epsilon *= 0.995  # Slower decay

            # Save model every 10 episodes
            if random.random() < 1 / self.__class__.target_update_freq:
                        self.__class__.target_model.load_state_dict(self.__class__.model.state_dict())
                        print("Target network updated!")

            if random.random() < 0.1:  # Approx every 10 episodes
                torch.save(self.__class__.model.state_dict(), "dqn_bomberman.pth")
                print("Model saved!")
    
    def wall_blocked(self, wrld):
        """
        Checks if there is a full horizontal wall line (excluding the outer boundaries)
        that spans the width of the world.
        
        Returns:
            True if at least one row (from y = 1 to y = wrld.height()-2) is completely 
            filled with walls (for x in range(1, wrld.width()-1)), otherwise False.
        """
        for y in range(1, wrld.height() - 1):
            if all(wrld.wall_at(x, y) for x in range(1, wrld.width() - 1)):
                return True
        return False

    # --- Main decision method (called each turn) ---
    def do(self, wrld):
        # Get current state.
        state = self.get_state(wrld)
        
        # If we have a recorded previous state-action, form a transition.
        if self.last_state is not None:
            reward, done = self.get_reward(wrld, self.last_state, state)
            self.remember(self.last_state, self.last_action, reward, state, done)
            self.train_model()
            # If the episode is done (exit reached or explosion hit), reset our saved state.
            if done:
                self.last_state = None
                self.last_action = None
                return
        
        # Epsilon-greedy policy: choose a random action with probability epsilon.
        if random.random() < self.__class__.epsilon:
            action = random.randint(0, self.__class__.output_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state).unsqueeze(0)
                q_vals = self.__class__.model(state_tensor)
                action = torch.argmax(q_vals).item()
        
        # Save state and action for the next transition update.
        self.last_state = state
        self.last_action = action
        
        # Map the chosen action to the in-game command.
        # Actions: 0: up, 1: down, 2: left, 3: right, 4: stay, 5: bomb.
        if action == 0:
            self.move(0, -1)
        elif action == 1:
            self.move(0, 1)
        elif action == 2:
            self.move(-1, 0)
        elif action == 3:
            self.move(1, 0)
        elif action == 4:
            self.move(0, 0)
        elif action == 5:
            self.place_bomb()