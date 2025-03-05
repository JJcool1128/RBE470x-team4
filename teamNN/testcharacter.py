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
    input_dim = 9    # Size of our state vector (see get_state below)
    output_dim = 6   # Number of actions (up, down, left, right, stay, bomb)
    # Our Q-network is defined as a class variable so that its memory and training
    # persist across episodes.
    # Our main Q-network
    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=2000)

    # # Add Target Network (to stabilize learning)
    # target_model = DQN(input_dim, output_dim)
    # target_model.load_state_dict(model.state_dict())  # Copy initial weights
    # target_model.eval()  # Target network is only used for computing Q-targets

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
        # try:
        #     self.__class__.target_model.load_state_dict(self.__class__.model.state_dict())
        #     # self.__class__.model.eval()  # Switch to evaluation mode
        #     # self.__class__.epsilon = 0.05  # Reduce randomness for deployment
        #     print("Loaded trained model successfully!")
        # except FileNotFoundError:
        #     print("No pre-trained model found. Training from scratch.")

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
            monster_dx = -1.0
            monster_dy = -1.0
            monster_dist = -1.0  # default value when no monster is nearby
        else:
            monster_dx = (closest_monster.x - self.x) / width
            monster_dy = (closest_monster.y - self.y) / height
            monster_dist = min_dist / (width + height)

        bomb_x = -1.0
        bomb_y = -1.0
        for dx in range(wrld.width() - 1):
            for dy in range(wrld.height() - 1):
                if wrld.bomb_at(dx,dy):
                    bomb_x = dx / width
                    bomb_y = dy / height
                    break

        # explosion_pos_list = []
        # for dx in range(wrld.width() - 1):
        #     for dy in range(wrld.height() - 1):
        #         if wrld.explosion_at(dx,dy):
        #             explosion_x = dx / width
        #             explosion_y = dy / height
        #             explosion_pos_list.append((explosion_x, explosion_y))
        #         else:
        #             explosion_x = -1.0
        #             explosion_y = -1.0

        state = np.array([agent_x, agent_y, exit_x, exit_y,
                          monster_dx, monster_dy, monster_dist, bomb_x, bomb_y],
                         dtype=np.float32)
        return state

    # --- Reward function ---
    def get_reward(self, wrld, state, next_state, action):
        """
        A basic reward function:
        """
        reward = 0
        done = False

        print("Agent x: ", str(next_state[0]*wrld.width()))
        print("Agent y: ", str(next_state[1]*wrld.height()))
        print("Exit x: ", str(next_state[2]*wrld.width()))
        print("Exit y: ", str(next_state[3]*wrld.height()))
        print("Monster dx: ", str(next_state[4]*wrld.width()))
        print("Monster dy: ", str(next_state[5]*wrld.height()))
        print("Monster dist: ", str(next_state[6]))
        print("Bomb x: ", str(next_state[7]*wrld.width()))
        print("Bomb y: ", str(next_state[8]*wrld.height()))
        
        # # Reward for reaching the exit.
        # if state[0] == state[2] and state[1] == state[3]:
        #     reward += 10.0
        #     done = True
        #     print("Reached goal: TRUE")
        # else:
        #     print("Reached goal: FALSE")

        # # Penalty if the agent is hit by an explosion.
        # if state[0] == state[7] and state[1] == state[8]:
        #     reward -= 10.0
        #     done = True
        #     print("Hit by explosion: TRUE")
        # else:
        #     print("Hit by explosion: FALSE")

        # if action == 5:
        #     print("Placed bomb: TRUE")
        #     if next_state[0] == next_state[7] and next_state[1] == next_state[8]:
        #         reward -= 1
        #         print("After Placing Bomb, same location as bomb: TRUE")
        #     else: 
        #         print("After Placing Bomb, same location as bomb: FALSE")
        # else:
        #     print("Placed bomb: FALSE")
            
        # if abs(state[0] - state[7]) == 1/wrld.width() or abs(state[1] - state[8]) == 1/wrld.height():
        #     print("Within bomb range == TRUE")
        #     if action == 4:
        #         reward -= 1
        #         print("Moved away within bomb range: FALSE")
        #     else:
        #         print("Moved away within bomb range: TRUE")
            
        #     if next_state[0] == next_state[7] and next_state[1] == next_state[8]:
        #         reward -= 1
        #         print("Went back to bomb: TRUE")
        #     else:
        #         print("Went back to bomb: FALSE")
        # else:
        #     print("Within bomb range: FALSE")

        # # Extra penalty if a monster is dangerously close.
        # if abs(next_state[4]) <= 2/wrld.width() or abs(next_state[5]) <= 2/wrld.height():
        #     reward -= 2.0
        #     print("Monster close: TRUE")
        # else: 
        #     print("Monster close: FALSE \n")

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

            # # Compute Q-values and targets
            # q_values = self.__class__.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            # next_q_values = self.__class__.target_model(next_states).max(1)[0]
            # target = rewards + self.__class__.gamma * next_q_values * (~dones)

            # Compute loss and optimize
            # loss = self.__class__.loss_fn(q_values, target.detach())
            # self.__class__.optimizer.zero_grad()
            # loss.backward()
            # self.__class__.optimizer.step()

            # Decay epsilon
            # Only decay epsilon once per episode, not per step
            if len(self.__class__.memory) >= self.__class__.batch_size:
                if self.__class__.epsilon > self.__class__.epsilon_min:
                    self.__class__.epsilon *= 0.995  # Slower decay

            # Save model every 10 episodes
            # if random.random() < 1 / self.__class__.target_update_freq:
            #     self.__class__.target_model.load_state_dict(self.__class__.model.state_dict())
            #     print("Target network updated!")

            # if random.random() < 0.1:  # Approx every 10 episodes
            #     torch.save(self.__class__.model.state_dict(), "dqn_bomberman.pth")
            #     print("Model saved!")
    
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
            reward, done = self.get_reward(wrld, self.last_state, state, self.last_action)
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