import sys
sys.path.insert(0, '../bomberman')
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from math import sqrt
from game import Game
from enum import Enum   
from random import randint
from events import Event
from sensed_world import SensedWorld
from numpy import array
import numpy as np

class State(Enum):
    STAY_IN_PLACE = 0
    ESCAPING = 1
    MONSTER_NEAR_BY = 2
    WALL_NEAR_BY = 3
    EXIT_NEAR_BY = 4
    AGENT_DEAD = 5
    KILLED_MONSTER = 6
    AVOID_EXPLOSION = 7 
    CHARACTER_EXITED = 8
    PLACING_BOMB = 9
    NAVIGATING = 10
    
class TestCharacter(CharacterEntity):
    print("TestCharacter class loaded!")

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.escaping = False   
        self.dead = False
        self.exited = False
        self.killed_monster_recently = False
        self.placing_bomb = False
        self.navigating = False
        self.weights = {
            "bias": 0.0,
            "distance_to_exit": 0.0,
            "distance_to_monster": 0.0,
            "bomb_risk": 0.0,
            "action": 0.0
        }
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1 # Exploration rate
        actions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        for action in actions:
            self.weights[f'action_{action}'] = 0.0 

    def updateState(self, wrld) -> State:
        # Check if agent is dead
        if self.is_dead(wrld):
            print("State check: AGENT_DEAD")
            return State.AGENT_DEAD

        # Check if the exit is immediately adjacent
        exit_cell = (wrld.exitcell[0], wrld.exitcell[1])
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = self.x + dx, self.y + dy
            print(f"Checking cell {(nx, ny)} for exit: {(nx, ny) == exit_cell}")
            if (nx, ny) == exit_cell:
                print("State check: EXIT_NEAR_BY")
                return State.EXIT_NEAR_BY

        # Check for nearby monsters (2 < distance < 4 => MONSTER_NEAR_BY, distance < 2 => ESCAPING)
        from math import sqrt
        for monster_list in wrld.monsters.values():
            for m in monster_list:
                dist = sqrt((self.x - m.x)**2 + (self.y - m.y)**2)
                print(f"Distance to monster at {(m.x, m.y)}: {dist}")
                if dist < 4:
                    if dist < 2:
                        print("State check: ESCAPING")
                        return State.ESCAPING
                    else:
                        print("State check: MONSTER_NEAR_BY")
                        return State.MONSTER_NEAR_BY

        # If we’re in the process of placing a bomb, return that state
        if self.placing_bomb:
            print("State check: PLACING_BOMB")
            return State.PLACING_BOMB

        # Check for walls in adjacent cells => WALL_NEAR_BY
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                if wrld.wall_at(nx, ny):
                    print("State check: WALL_NEAR_BY")
                    return State.WALL_NEAR_BY

        # Default => STAY_IN_PLACE
        print("State check: STAY_IN_PLACE (default)")
        return State.STAY_IN_PLACE

    # Process game events to update flags
    def process_events(self, events):
        for e in events:
            if e.tpe == Event.BOMB_HIT_MONSTER:
                print(f"{self.name}'s bomb killed a monster: {e.other.name}")
                self.killed_monster_recently = True
            elif e.tpe == Event.CHARACTER_FOUND_EXIT:
                # e.character is the character who found the exit.
                if e.character == self:
                    print(f"{self.name} found the exit!")
                    self.exited = True
            elif e.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                # e.character is the killed character.
                if e.character == self:
                    print(f"{self.name} was killed by a monster!")
                    self.dead = True

    # Main decision method called each turn
    def do(self, wrld):
        # Clone to sensed world, process events
        sensed_wrld = SensedWorld.from_world(wrld)
        (next_sensed, events) = sensed_wrld.next()
        self.process_events(events)

         # Check current state
        current_state = self.updateState(next_sensed)
        print(f"Current state: {current_state.name}")
        
        start = (self.x, self.y)
        goal = (wrld.exitcell[0], wrld.exitcell[1])
        # Retrieve events from the world (this depends on your implementation)
        (new_wrld, events) = wrld.next()  # adjust if necessary
        self.process_events(events)

        if current_state == State.AVOID_EXPLOSION:
            self.evade_bomb(wrld)
            print("Acting on AVOID_EXPLOSION state")
            return

        # React based on the current state
        if current_state == State.PLACING_BOMB:
            print("Acting on PLACING_BOMB state")
            #self.kill_monster(wrld)
            self.placing_bomb = False
            return

        elif current_state == State.NAVIGATING:
            print("Acting on NAVIGATING state")
            self.navigating = False  
            return

        elif current_state == State.ESCAPING:
            print("Acting on ESCAPING state")
            self.escape_monster(wrld)
            return

        elif current_state == State.MONSTER_NEAR_BY:
            print("Acting on MONSTER_NEAR_BY state: navigating carefully")
            self.navigate_carefully(wrld)
            return

        elif current_state == State.EXIT_NEAR_BY:
            print("Acting on EXIT_NEAR_BY state: moving toward exit")
            self.follow_path(wrld)
            return

        elif current_state == State.AGENT_DEAD:
            print("Agent is dead. Taking no action.")
            self.move(0, 0)
            return
        
        bomb_distance = self.get_dist_to_bomb(wrld)
        if bomb_distance <= 2:
            return State.AVOID_EXPLOSION

        self.escaping = False
        action = self.get_best_action(wrld)
        dx, dy = action
        self.move(dx, dy)
        
        next_state = wrld.next()  # Get next state after move
        reward = self.get_reward(new_wrld, self.actions_near_wall(wrld))
        self.update_weights(new_wrld, action, reward, new_wrld)

        # # Check for nearby monsters; if detected, use minimax escape strategy.
        # for monster in wrld.monsters.values():
        #     for m in monster:
        #         mx, my = m.x, m.y
        #         monster_distance = float(sqrt(abs(self.x - mx)**2 + abs(self.y - my)**2))
        #         if monster_distance < 4:
        #             print(f"Monster detected {monster_distance} tiles away! Switching to minimax escape!")
        #             self.escaping = True
        #             self.escape_monster(wrld)
        #             return  

    # Get non-wall neighboring positions (up, down, left, right)
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
    
    def evade_bomb(self, wrld):
   
        print("Evading bomb!")

        best_move = (0, 0)  # Default: Stay in place
        max_dist = self.get_dist_to_bomb(wrld)  # Current distance

        for move in self.neighbors_of_4(wrld, (self.x, self.y)): 
            move_x, move_y = move
            new_dist = abs(move_x - self.x) + abs(move_y - self.y) 

        if new_dist > max_dist: 
            best_move = (move_x - self.x, move_y - self.y)
            max_dist = new_dist

        print(f"Evading bomb, moving to: {best_move}")
        self.move(best_move[0], best_move[1])

    # Manhattan distance heuristic for A*
    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # Calculate cost of moving from point a to b considering hazards
    def cost(self, wrld, a: tuple[int, int], b: tuple[int, int], explosion_cells, current_time) -> int:
        ax, ay = a
        bx, by = b
        final_cost = sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        if wrld.monsters_at(bx, by):
            return float('inf')
        for dx in [-1, 0, 1]:
            if wrld.bomb_at(bx+dx, by):
                return float('inf')
        for dy in [-1, 0, 1]:
            if wrld.bomb_at(bx, by+dy):
                return float('inf')
        if wrld.explosion_at(bx, by):
            return float('inf')
        return final_cost

    # Follow the A* path towards the exit
    def follow_path(self, wrld):
        start = (self.x, self.y)
        goal = (wrld.exitcell[0], wrld.exitcell[1])
        path = self.astar(wrld, start, goal, explosion_cells={}, current_time=wrld.time)
        if not path or len(path) < 2:  
            print("No valid path found, staying in place.")
            self.move(0, 0)
            return
        next_x, next_y = path[1]
        dx, dy = next_x - start[0], next_y - start[1]
        self.move(dx, dy)

    # Try to navigate carefully by choosing safe neighboring moves
    def navigate_carefully(self, wrld):
        print("Navigating carefully to avoid monsters...")
        monsters = [(m.x, m.y) for sublist in wrld.monsters.values() for m in sublist]
        start = (self.x, self.y)
        safe_moves = []
        for move in self.neighbors_of_4(wrld, start):
            if all(abs(move[0] - mx) + abs(move[1] - my) > 2 for mx, my in monsters):  
                safe_moves.append(move)
        if safe_moves:
            best_move = max(safe_moves, key=lambda m: len(self.neighbors_of_4(wrld, m)))
            dx, dy = best_move[0] - start[0], best_move[1] - start[1]
            print(f"Moving safely to {best_move}")
            self.move(dx, dy)
            return
        self.follow_path(wrld)

    # Use minimax strategy to escape a nearby monster with explosion avoidance
    def escape_monster(self, wrld):
        print("ESCAPING using minimax with explosion avoidance!")
        closest_monster = None
        min_distance = float('inf')
        for monster in wrld.monsters.values():
            for m in monster:
                d = abs(self.x - m.x) + abs(self.y - m.y)
                if d < min_distance:
                    min_distance = d
                    closest_monster = m
        if closest_monster is None:
            print("No monster found. Proceeding normally.")
            return
        current_pos = (self.x, self.y)
        candidate_moves = self.neighbors_of_4(wrld, current_pos)
        candidate_moves.append(current_pos) 
        safe_moves = []
        for move in candidate_moves:
            x, y = move
            if wrld.bomb_at(x, y) or wrld.bomb_at(x+1, y) or wrld.bomb_at(x, y+1) or wrld.bomb_at(x-1, y) or wrld.bomb_at(x, y-1) or wrld.explosion_at(x, y):
                print(f"Skipping move {move} because it has a bomb or explosion.")
                continue
            safe_moves.append(move)
        if not safe_moves:
            print("No safe moves available! Staying in place.")
            self.move(0, 0)
            return
        best_move = current_pos
        best_value = -float('inf')
        for move in safe_moves:
            value = self.minimax(
                wrld,
                char_pos=move,
                monster_pos=(closest_monster.x, closest_monster.y),
                depth=2,
                is_maximizing=False
            )
            print(f"Evaluated move {move} with minimax value {value}")
            if value > best_value:
                best_value = value
                best_move = move
        dx, dy = best_move[0] - self.x, best_move[1] - self.y
        print(f"Minimax selected move {best_move} with value {best_value}")
        for monster in wrld.monsters.values():
            for m in monster:
                monster_distance = float(sqrt(abs(self.x - m.x)**2 + abs(self.y - m.y)**2))
                if monster_distance < 3:
                    if not (wrld.bomb_at(m.x, m.y) or wrld.bomb_at(m.x+1, m.y) or wrld.bomb_at(m.x, m.y+1) or wrld.bomb_at(m.x-1, m.y) or wrld.bomb_at(m.x, m.y-1)):
                        self.place_bomb()
        self.move(dx, dy)

    # Minimax algorithm for decision making during escape
    def minimax(self, wrld, char_pos, monster_pos, depth, is_maximizing):
        if depth == 0:
            return abs(char_pos[0] - monster_pos[0]) + abs(char_pos[1] - monster_pos[1])
        if is_maximizing:
            best_val = -float('inf')
            moves = self.neighbors_of_4(wrld, char_pos)
            moves.append(char_pos)
            for move in moves:
                if not (0 <= move[0] < wrld.width() and 0 <= move[1] < wrld.height()):
                    continue
                if wrld.wall_at(move[0], move[1]):
                    continue
                val = self.minimax(wrld, move, monster_pos, depth - 1, False)
                best_val = max(best_val, val)
            return best_val
        else:
            best_val = float('inf')
            moves = self.monster_neighbors(wrld, monster_pos)
            for move in moves:
                if not (0 <= move[0] < wrld.width() and 0 <= move[1] < wrld.height()):
                    continue
                if wrld.wall_at(move[0], move[1]):
                    continue
                val = self.minimax(wrld, char_pos, move, depth - 1, True)
                best_val = min(best_val, val)
            return best_val

    # Returns valid moves for the monster (including diagonal moves)
    def monster_neighbors(self, wrld, pos: tuple[int, int]) -> list[tuple[int, int]]:
        moves = []
        x, y = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                    if not wrld.wall_at(nx, ny):
                        moves.append((nx, ny))
        return moves

    # Checks to see if the agent died by monster or by suicide 
    def is_dead(self, wrld) -> bool:
        if wrld.explosion_at(self.x, self.y):
            print("Agent was hit by a bomb explosion!")
            return True

        from math import sqrt
        for monster in wrld.monsters.values():
            for m in monster:
                distance = sqrt((self.x - m.x)**2 + (self.y - m.y)**2)
                # Adjust threshold: here, distance less than 1.0 is lethal.
                if distance < 1.0:
                    print("Agent was killed by a monster!")
                    return True
        return False

    # APPROXIMATE Q-LEARNING ALGORITHM

    def get_dist_to_monster(self, wrld):
        monsters = [(m.x, m.y) for sublist in wrld.monsters.values() for m in sublist]
        distances = [abs(self.x - mx) + abs(self.y - my) for mx, my in monsters]
        if not distances:
            return 100
        return min(distances) 
    
    def get_dist_to_exit(self, wrld):
        goal = (wrld.exitcell[0], wrld.exitcell[1])
        return abs(self.x - goal[0]) + abs(self.y - goal[1])
    
    def get_dist_to_bomb(self, wrld):
        min_dist = float('inf') 

        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y):
                    bomb_dist = abs(self.x - x) + abs(self.y - y)  
                    min_dist = min(min_dist, bomb_dist)  

        return min_dist if min_dist != float('inf') else 100

    def state_features(self, wrld, action) -> dict:
        # Returns a dictionary of features for the current state and action
        features = {
            "bias": 1.0,
            "distance_to_monster": self.get_dist_to_monster(wrld),  # changed key name
            "distance_to_exit": self.get_dist_to_exit(wrld),          # key already matches
            "bomb_risk": 1.0 if wrld.explosion_at(self.x, self.y) or wrld.bomb_at(self.x, self.y) else 0.0,
            "action_x": action[0],
            "action_y": action[1]
        }
        return features
    
    def get_q_value(self, wrld, action):
        action_key = f'action_{action}'
        #return self.weights.get(action_key, 0.0) 
        features = self.state_features(wrld, action)
        #return sum(self.weights[f] * features[f] for f in features)
        return sum(self.weights.get(f, 0.0) * features[f] for f in features)
    

    def get_best_action(self, wrld):
        actions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # Stay, move up, down, right, left
        if randint(0, 100) < self.epsilon * 100:  # Explore
            return actions[randint(0, len(actions) - 1)]
        
        q_values = [(action, self.get_q_value(wrld, action)) for action in actions]
        max_q = max(q_values, key=lambda x: x[1])[1]

        best_actions = [action for action, q in q_values if q == max_q]
    
        return best_actions[randint(0, len(best_actions) - 1)]
        
        

    def update_weights(self, wrld, action, reward, next_state):
        """
        Perform the Q-learning weight update.
        """
        features = self.state_features(wrld, action)
        q_value = self.get_q_value(wrld, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in [(0,0), (0,1), (0,-1), (1,0), (-1,0)]])
        
        target = reward + self.gamma * max_next_q
        error = target - q_value

        # Update weights
        for f in features:
            if f not in self.weights:
                self.weights[f] = 0.0
            self.weights[f] += self.alpha * error * features[f]
            #print(f"Updated weights: {self.weights}")

    def actions_near_wall(self, wrld):
        place_bomb = False
        wall_action = np.array([place_bomb],dtype=bool)
        # if all(wrld.wall_at(self.x + dx, self.y + dy) for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]) and State.WALL_NEAR_BY:
        #     place_bomb = True
        #     self.updateState(wrld) == State.PLACING_BOMB
        #     self.place_bomb()
        wall_count = sum(1 for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)] if wrld.wall_at(self.x + dx, self.y + dy))

    # Place a bomb if adjacent to at least one wall, but not completely trapped
        if 1 <= wall_count <= 2:  
            place_bomb = True
            self.placing_bomb = True  # Track that a bomb is placed
            self.place_bomb()
        return wall_action


    def get_reward(self, wrld, wall_action):
        """
        Reward function based on state.
        """
        reward = 0

        if wrld.explosion_at(self.x, self.y):  # Died
            reward -= 100
        elif (self.x, self.y) == wrld.exitcell:  # Reached exit
            reward += 100
        elif any(m.x == self.x and m.y == self.y for sublist in wrld.monsters.values() for m in sublist):  # Caught by monster
            reward -= 50
        if wall_action[0]: 
            reward += 50
            print("Bomb placed near wall! Reward given!")
        
        bomb_distance = self.get_dist_to_bomb(wrld)
        if bomb_distance <= 2:  # If within explosion range
            penalty = (3 - bomb_distance) * 5  # Stronger penalty for closer distance
            reward -= penalty

        reward -= 1  
        print(f"Reward: {reward}, Bomb Placed: {wall_action[0]}") 
        return reward

    # avoiding bomb - manhattan dist btw agent and bomb, update 

    # Attempt to kill a monster by placing a bomb and escaping
    # def kill_monster(self, wrld):
    #     self.placing_bomb = True
    #     current_x, current_y = self.x, self.y
    #     explosion_cells = {}
    #     explosion_range = wrld.expl_range  
    #     explosion_time = wrld.time + 1  
    #     explosion_duration = wrld.expl_duration  
    #     for d in range(-explosion_range, explosion_range + 1):
    #         if 0 <= current_x + d < wrld.width():
    #             explosion_cells[(current_x + d, current_y)] = explosion_time + explosion_duration
    #         if 0 <= current_y + d < wrld.height():
    #             explosion_cells[(current_x, current_y + d)] = explosion_time + explosion_duration
    #     safe_moves = []
    #     for move in self.neighbors_of_4(wrld, (current_x, current_y)):
    #         if move not in explosion_cells:
    #             safe_moves.append(move)
    #     if safe_moves:
    #         best_move = max(safe_moves, key=lambda m: self.heuristic(m, (wrld.exitcell[0], wrld.exitcell[1])))
    #         dx, dy = best_move[0] - current_x, best_move[1] - current_y
    #         print(f" Fast escape to {best_move}, avoiding explosion!")
    #         self.place_bomb()
    #         self.move(dx, dy)
    #         return 
    #     print(" No immediate safe move. Using A* for escape.")
    #     safe_path = self.astar(wrld, (current_x, current_y), (wrld.exitcell[0], wrld.exitcell[1]), explosion_cells, wrld.time)
    #     if safe_path and len(safe_path) > 2:
    #         print(" Found a longer escape path!")
    #         self.place_bomb()
    #         next_x, next_y = safe_path[1]
    #         dx, dy = next_x - current_x, next_y - current_y
    #         self.move(dx, dy)
    #     else:
    #         print(" No safe escape! Bomb placement canceled.")