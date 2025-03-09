import sys
sys.path.insert(0, '../bomberman')
from entity import CharacterEntity
from events import Event
import random
import numpy as np
from collections import deque
from sensed_world import SensedWorld
from events import Event
import time


# Import PyTorch for the neural network implementation.
import torch
import torch.nn as nn
import torch.optim as opti

# Defines the DQN neural network. This network takes a state vector 
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
    
    input_dim = 10    # Size of our state vector (see get_state below)
    output_dim = 5   # Number of actions (up, down, left, right, stay, bomb)

    # DQN Hyperparameters
    gamma = 0.995
    epsilon = 1.0       # Initial exploration rate
    epsilon_min = 0.2
    epsilon_decay = 0.99
    batch_size = 32
    target_update_freq = 100  # Update target network every 50 episodes

    # We'll keep a global step counter to know when to do updates
    global_training_steps = 0

    # Our Q-network is defined as a class variable so that its memory and training
    # persist across episodes.
    # Our main Q-network
    model = DQN(input_dim, output_dim)
    optimizer = opti.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()

    # Add Target Network for Double DQN
    target_model = DQN(input_dim, output_dim)

    memory = deque(maxlen=2000)

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        # To record the previous state and action for the transition update.
        self.last_state = None
        self.last_action = None

        self.won = False

        self.bomb_exists = False

        self.crossed_levels = {
            4:  False,
            8:  False,
            12: False,
            16: False
        }

        try:
            self.__class__.model.load_state_dict(torch.load("dqn_bomberman.pth"))
            self.__class__.model.eval()
            print("Model loaded successfully!")
        except FileNotFoundError:
            # If the file doesn't exist yet, just continue with random initialization
            print("Model file not found. Continuing with a randomly initialized network.")

    def get_state(self, wrld):

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

        explosion_at_agent = 1.0 if wrld.explosion_at(self.x, self.y) else 0.0

        state = np.array([agent_x, agent_y, exit_x, exit_y,
                          monster_dx, monster_dy, monster_dist, bomb_x, bomb_y, explosion_at_agent],
                         dtype=np.float32)
        return state

    def get_reward(self, old_state, next_state, events, new_wrld, action):
        reward = -0.1
        done = False

        # print("Last Agent x: ", str(round(old_state[0]*new_wrld.width())), "    Current Agent x: ", str(round(next_state[0]*new_wrld.width())) )
        # print("Last Agent y: ", str(round(old_state[1]*new_wrld.height())), "   Current Agent y: ", str(round(next_state[1]*new_wrld.height())) )
        # print("Last Exit x: ", str(round(old_state[2]*new_wrld.width())), "    Current Exit x: ", str(round(next_state[2]*new_wrld.width())) )
        # print("Last Exit y: ", str(round(old_state[3]*new_wrld.height())), "    Current Exit y: ", str(round(next_state[3]*new_wrld.height())) )
        # print("Last Monster dx: ", str(round(old_state[4]*new_wrld.width())), "    Current Monster dx: ", str(round(next_state[4]*new_wrld.width())))
        # print("Last Monster dy: ", str(round(old_state[5]*new_wrld.height())), "    Current Monster dy: ", str(round(next_state[5]*new_wrld.height())))
        # print("Last Bomb x: ", str(round(old_state[7]*new_wrld.width())), "    Current Bomb x: ", str(round(next_state[7]*new_wrld.width())))
        # print("Last Bomb y: ", str(round(old_state[8]*new_wrld.height())), "    Current Bomb y: ", str(round(next_state[8]*new_wrld.height())))

        me_in_clone = new_wrld.me(self)

        for ev in events:
            # REACHED EXIT?
            if (ev.tpe == Event.CHARACTER_FOUND_EXIT 
                # and me_in_clone is not None
                # and ev.character is not None 
                # and ev.character.name == me_in_clone.name
                ):
                reward += 200.0
                print("Character Found EXIT (it was me_in_clone!)")
                done = True
                self.won = True

            # KILLED BY MONSTER?
            if (ev.tpe == Event.CHARACTER_KILLED_BY_MONSTER
                # and me_in_clone is not None
                # and ev.character is not None
                # and ev.character.name == me_in_clone.name
                ):
                reward -= 50.0
                print("Character Killed by MONSTER (it was me!)")
                done = True

            # HIT BY BOMB EXPLOSION?
            if (ev.tpe == Event.BOMB_HIT_CHARACTER
                # and me_in_clone is not None
                # and ev.other is not None
                # and ev.other.name == me_in_clone.name
                ):
                reward -= 50.0
                print("Character Killed by BOMB (it was me!)")
                done = True
            
            if (ev.tpe == Event.BOMB_HIT_MONSTER
                # and me_in_clone is not None
                # and ev.other is not None
                # and ev.other.name == me_in_clone.name
                ):
                reward += 50.0
                print("MONSTER Killed by BOMB (it was me!)")
                done = True

            if ev.tpe == Event.BOMB_HIT_WALL:
                reward += 5.0  
                print("Bomb destroyed a wall!")

        # 2) Extra shaping: did we move closer to the exit?
        old_dist_to_exit = abs(round(old_state[0]*new_wrld.width()) - round(old_state[2]*new_wrld.width())) \
                         + abs(round(old_state[1]*new_wrld.height()) - round(old_state[3]*new_wrld.height()))
        new_dist_to_exit = abs(round(next_state[0]*new_wrld.width()) - round(next_state[2]*new_wrld.width())) \
                         + abs(round(next_state[1]*new_wrld.height()) - round(next_state[3]*new_wrld.height()))
        if new_dist_to_exit < old_dist_to_exit:
            reward += 1
        elif new_dist_to_exit > old_dist_to_exit:
            reward -= 1
        
        # Penalty for placing a bomb for no reason
        if not self.bomb_exists:
            if action == 4:
                if round(old_state[1]*new_wrld.height()) < new_wrld.height() - 1:
                    if new_wrld.wall_at(round(old_state[0]*new_wrld.width()),round(old_state[1]*new_wrld.height())+1):
                        reward += 1
                        # print("Bomb placed near wall: TRUE")
                    elif not (abs(round(old_state[4]*new_wrld.width())) + abs(round(old_state[5]*new_wrld.height()))) < 4:
                        reward -= 1
                        # print("Bomb placed for no reason: TRUE")
                    # else:
                    #     print("Bomb placed for no reason: FALSE")
                elif not abs(round(old_state[4]*new_wrld.width())) + abs(round(old_state[5]*new_wrld.height())) < 4:
                    reward -= 1
                    # print("Bomb placed for no reason: TRUE")
                else:
                    reward += 1
                    # print("Bomb placed for no reason: FALSE")

        # Penalty for being close to a monster
        if (np.sqrt((round(old_state[4]*new_wrld.width())**2 + round(old_state[5]*new_wrld.height())**2))) < 3:
            # print("Agent was close to monster: TRUE")
            if (np.sqrt((round(next_state[4]*new_wrld.width())**2) + np.sqrt(round(next_state[5]*new_wrld.height())**2))) < (np.sqrt((round(old_state[4]*new_wrld.width())**2 + round(old_state[5]*new_wrld.height())**2))):
                reward -= 1
                # print("Agent was close to monster and went closer: TRUE")
            else:
                reward += 1
                # print("Agent was close to monster and went closer: FALSE")
        # else:
        #     print("Agent was close to monster: FALSE")

        # Penalty for doing redundant moves
        if (round(old_state[0]*new_wrld.width()) == 0 and action == 2):
            reward -= 0.2
            # print("Agent tried to move into an unreachable cell: LEFT")
        if (round(old_state[0]*new_wrld.width()) == new_wrld.width()-1 and action == 3):
            reward -= 0.2
            # print("Agent tried to move into an unreachable cell: RIGHT")
        if (round(old_state[1]*new_wrld.height()) == 0 and action == 0):
            reward -= 0.2
            # print("Agent tried to move into an unreachable cell: UP")
        if (round(old_state[1]*new_wrld.height()) == new_wrld.height()-1 and action == 1):
            reward -= 0.2
            # print("Agent tried to move into an unreachable cell: DOWN")
        if (self.bomb_exists and action ==4):
            reward -= 0.2
            # print("Agent tried to place a bomb with a bomb already there: TRUE")
        if round(old_state[1]*new_wrld.height()) < new_wrld.height() - 1:
            if (new_wrld.wall_at(round(old_state[0]*new_wrld.width()),round(old_state[1]*new_wrld.height())+1) and action == 1):
                reward -= 0.2
                # print("Agent tried to move into a wall: DOWN")

        # checking the world for bombs
        self.bomb_exists = False
        for dx in range(new_wrld.width()-1):
            for dy in range(new_wrld.height()-1):
                if new_wrld.bomb_at(dx, dy):
                    self.bomb_exists = True 
                    break
            if self.bomb_exists:
                break

        # Reward for crossing each level of walls
        old_y = round(old_state[1] * new_wrld.height())
        new_y = round(next_state[1] * new_wrld.height())

        for threshold in [4, 8, 12, 16]:
            if old_y < threshold <= new_y and not self.crossed_levels[threshold]:
                reward += 15.0  
                self.crossed_levels[threshold] = True
                # print(f"Crossed level {threshold} for the first time!")
                        
        # print(reward)
        return reward, done

    def remember(self, state, action, reward, next_state, done):
        self.__class__.memory.append((state, action, reward, next_state, done, self.won))

    def train_model(self):
        if len(self.__class__.memory) < self.__class__.batch_size:
            return
        batch = random.sample(self.__class__.memory, self.__class__.batch_size)
        states, actions, rewards, next_states, dones, wons = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        wons = torch.tensor(wons, dtype=torch.bool)


        # Q(s_t, a_t) from current model
        q_values = self.__class__.model(states)              # shape: [batch_size, output_dim]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN logic:
        next_action_online = self.__class__.model(next_states).argmax(1)  # shape: [batch_size]
        next_q_vals_target = self.__class__.target_model(next_states)
        target_for_next_state = next_q_vals_target.gather(1, next_action_online.unsqueeze(1)).squeeze(1)

        # 3) If done, no future reward; else gamma * target_for_next_state
        not_done_mask = ~dones
        target = rewards + self.__class__.gamma * target_for_next_state * not_done_mask

        #Compute MSE loss
        loss = self.__class__.loss_fn(q_values, target.detach())

        # Optimize
        self.__class__.optimizer.zero_grad()
        loss.backward()
        self.__class__.optimizer.step()

        # Update global step and maybe update target network
        self.__class__.global_training_steps += 1
        if self.__class__.global_training_steps % self.__class__.target_update_freq == 0:
            # Copy weights from model to target_model
            self.__class__.target_model.load_state_dict(self.__class__.model.state_dict())

        # Epsilon decay
        if self.__class__.epsilon > self.__class__.epsilon_min:
            self.__class__.epsilon *= self.__class__.epsilon_decay  

        # Occasionally save model
        if random.random() < 0.1:
            torch.save(self.__class__.model.state_dict(), "dqn_bomberman.pth")


    def do(self, wrld):
        # Get the "current state" as your basis for picking an action
        current_state = self.get_state(wrld)

        # Epsilon-greedy action selection
        if random.random() < self.__class__.epsilon:
            action = random.randint(0, self.__class__.output_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(current_state).unsqueeze(0)
                q_vals = self.__class__.model(state_tensor)
                action = torch.argmax(q_vals).item()

        # SIMULATE that action in a cloned world
        temp_world = SensedWorld.from_world(wrld)
        me_in_clone = temp_world.me(self)

        if me_in_clone is not None:
            # Apply the chosen action to the clone
            if action == 0:   # move up
                me_in_clone.move(0, -1)
                # print("Action 0, Moved Up")
            elif action == 1: # move down
                me_in_clone.move(0, 1)
                # print("Action 1, Moved Down")
            elif action == 2: # move left
                me_in_clone.move(-1, 0)
                # print("Action 2, Moved Left")
            elif action == 3: # move right
                me_in_clone.move(1, 0)
                # print("Action 3, Moved Right")
            elif action == 4: # place bomb + stay
                me_in_clone.place_bomb()
                me_in_clone.move(0, 0)
                # print("Action 4, Placed Bomb")


        # Now step that cloned world forward to get the real next state + events
        new_world, events = temp_world.next()

        # Build a state vector from this new_world
        next_state = None
        me_after_step = new_world.me(self)
        if me_after_step is not None:
            old_real_x, old_real_y = self.x, self.y
            self.x, self.y = me_after_step.x, me_after_step.y
            next_state = self.get_state(new_world)
            # restore real x,y
            self.x, self.y = old_real_x, old_real_y
        else:
            next_state = current_state.copy()

        # Compute the reward for this transition
        reward, done = self.get_reward(
            old_state=current_state,
            next_state=next_state,
            events=events,
            new_wrld=new_world,
            action=action
        )

        # 6) Store in replay buffer
        self.remember(current_state, action, reward, next_state, done)
        self.train_model()

        # 7) If not done (didn't die or exit), actually perform the move in real wrld
        if action == 0:
            self.move(0, -1)
        elif action == 1:
            self.move(0, 1)
        elif action == 2:
            self.move(-1, 0)
        elif action == 3:
            self.move(1, 0)
        elif action == 4:
            self.place_bomb()
            self.move(0, 0)

        # 8) Save our last_state/last_action if you want to do 
        #    "start-of-next-turn" updating in the future, 
        #    but we've essentially already stored the transition right now.
        self.last_state = next_state
        self.last_action = action