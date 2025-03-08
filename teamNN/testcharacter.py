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
        self.prev_position = (x, y)
        self.bomb_cooldown = 0
        self.weights = {
            "bias": 0.0,
            "distance_to_exit": 0.0,
            "distance_to_monster": 0.0,
            "bomb_risk": 0.0,
            "action": 0.0
        }
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon_min = 0.1 # Exploration rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

        actions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        for action in actions:
            self.weights[f'action_{action}'] = 0.0 
    
    def update_epsilon(self):
        """Gradually decay epsilon from 1.0 to epsilon_min over 1000 iterations."""
        decay_rate = np.log(1.0 / self.epsilon_min) / 1000  # Decay over 1000 steps
        self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-decay_rate))  # Exponential decay
        print(f"Updated epsilon: {self.epsilon:.4f}")


    def updateState(self, wrld) -> State:

        if self.get_dist_to_bomb(wrld) <= 3:
            return State.AVOID_EXPLOSION

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

    def get_section_goal(self, wrld):
        section_bottom = wrld.height() - 1  # default: bottom of the world
        # Search for the next wall row starting from the row below the agent
        for row in range(self.y + 1, wrld.height()):
            # Count the number of wall cells in this row
            wall_count = sum(1 for x in range(wrld.width()) if wrld.wall_at(x, row))
            # If almost the entire row is a wall (adjust the threshold as needed), we are at a section boundary.
            if wall_count >= wrld.width() - 1:
                section_bottom = row - 1
                break

        # Now, pick the rightmost cell in that section that is not a wall.
        goal_x = wrld.width() - 1
        while goal_x >= 0 and wrld.wall_at(goal_x, section_bottom):
            goal_x -= 1
        return (goal_x, section_bottom)

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
            self.evade_bomb(wrld,exit_x=goal[0], exit_y=goal[1])
            print("Acting on AVOID_EXPLOSION state")
            return

        # React based on the current state
        if current_state == State.PLACING_BOMB:
            print("Acting on PLACING_BOMB state")
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

        self.escaping = False
        action = self.get_best_action(wrld)
        dx, dy = action
        self.move(dx, dy)
        
        next_state = wrld.next()  # Get next state after move
        reward = self.get_reward(new_wrld, self.actions_near_wall(new_wrld))
        self.update_weights(new_wrld, action, reward, new_wrld)
        self.prev_position = (self.x, self.y)  # Update previous position


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
    
    def get_dist_to_bomb_at(self, wrld, x, y):
        """
        Return the Manhattan distance from cell (x,y) to the nearest bomb,
        or 100 if no bombs in the grid.
        """
        min_dist = float('inf')
        for bx in range(wrld.width()):
            for by in range(wrld.height()):
                if wrld.bomb_at(bx, by):
                    dist = abs(x - bx) + abs(y - by)
                    if dist < min_dist:
                        min_dist = dist
        return min_dist if min_dist != float('inf') else 100
    
    def evade_bomb(self, wrld, exit_x, exit_y):

      best_dist = self.get_dist_to_bomb(wrld)  
      best_move = (0, 0)  # Default to staying in place
      possible_moves = []

      for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = self.x + dx, self.y + dy

        # Ignore out-of-bounds or walls
        if not (0 <= nx < wrld.width() and 0 <= ny < wrld.height()):
                continue
        if wrld.wall_at(nx, ny):
                continue
        if wrld.wall_at(nx, ny):
                continue
        if wrld.bomb_at(nx, ny):
                continue
        possible_moves.append((dx, dy))
        if possible_moves:
            best_move = possible_moves[randint(0, len(possible_moves) - 1)]
        self.move(best_move[0], best_move[1])

        # Move only if it increases the distance from bomb **and** moves toward exit
        new_dist = self.get_dist_to_bomb(wrld)
        exit_dist = abs(nx - exit_x) + abs(ny - exit_y)

        if new_dist > best_dist or exit_dist < abs(self.x - exit_x) + abs(self.y - exit_y):
            best_move = (dx, dy)
            best_dist = new_dist

        print(f"Moving {best_move} to avoid explosion and head toward exit!")
        self.move(best_move[0], best_move[1])

    def evade_bomb_monster_vector(self, wrld, monster_x, monster_y):
        steps_to_move = 2  # or as many steps as you want
        from math import copysign, sqrt

        for _ in range(steps_to_move):
            dx = self.x - monster_x  # monster->agent x-dist
            dy = self.y - monster_y  # monster->agent y-dist

            dist = sqrt(dx*dx + dy*dy) or 1  # avoid div by zero
            # unit vector away from monster
            unit_x = dx/dist
            unit_y = dy/dist

            # Round to nearest int direction
            move_x = 0
            if abs(unit_x) > 0.5:
                move_x = int(copysign(1, unit_x))  # +1 or -1

            move_y = 0
            if abs(unit_y) > 0.5:
                move_y = int(copysign(1, unit_y))

            # Attempt the move
            new_x = self.x + move_x
            new_y = self.y + move_y

            # If out of bounds or a wall, fallback or pick a simpler move
            if not (0 <= new_x < wrld.width() and 0 <= new_y < wrld.height()):
                print("Monster vector out-of-bounds, fallback to staying in place.")
                move_x, move_y = 0, 0
            elif wrld.wall_at(new_x, new_y):
                print("Wall encountered, fallback to staying in place.")
                move_x, move_y = 0, 0

            self.move(move_x, move_y)
            (temp_world, _) = wrld.next()
            if self.is_dead(temp_world):
                break

    def avoid_explosion_after_wall_bomb(self, wrld):
        steps_to_move = 2 

        for _ in range(steps_to_move):
            # Attempt to move up (0, -1)
            new_x = self.x
            new_y = self.y - 1

            # If out of bounds or is a wall, pick a fallback, e.g. (0,1)
            if not (0 <= new_x < wrld.width() and 0 <= new_y < wrld.height()):
                print("Can't move north (out of bounds). Try fallback (0,1).")
                new_y = self.y + 1  # fallback down

            if wrld.wall_at(new_x, new_y):
                print("Wall north, fallback to (0,1).")
                new_y = self.y + 1

            # Move that direction
            dy = new_y - self.y
            self.move(0, dy)
            # Because we're in a loop, we forcibly step the world
            # so that we actually see 2 separate moves in-game
            (temp_world, _) = wrld.next()
            # If agent died, break early
            if self.is_dead(temp_world):
                break

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
        goal = self.get_section_goal(wrld)
        wall_blocked, wall_level = self.wall_level(wrld)
        if wall_blocked:
            print(f"Wall detected at level {wall_level}! Evading...")
            # Adjust the goal if necessary (for instance, move to a cell just above the wall).
            goal = (goal[0], wall_level - 1)
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
    
    def wall_level(self, wrld):
        wall_blocked = False
        wall_level = 0
        count = 0
        for dy in range(wrld.height()-1):
            count = 0
            for dx in range(wrld.width()-1):
                if wrld.wall_at(dx, dy):
                    count += 1
                if count == 8:
                    wall_blocked = True
                    wall_level = dy
                    break
            if wall_blocked:
                break
        
        return wall_blocked, wall_level
    
    def astar(self, wrld, start: tuple[int, int], goal: tuple[int, int], explosion_cells, current_time) -> list[tuple[int, int]]:
        print("A* started...")
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        while not frontier.empty():
            _, current = frontier.get()
            if current == goal:
                print("Goal reached!")
                break
            for next in self.neighbors_of_4(wrld, current):
                if next in explosion_cells:
                    explosion_time = explosion_cells[next]
                    time_to_explode = explosion_time - current_time
                    if time_to_explode <= 2:
                        print(f"Explosion detected at {next} in {time_to_explode} turns!")
                        continue
                new_cost = cost_so_far[current] + self.cost(wrld, current, next, explosion_cells, current_time)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put((priority, next))
                    came_from[next] = current
        
        if goal not in came_from:
            print("No valid path to goal!")
            return []
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        print(f"Final path: {path}")
        return path

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
        found_bomb = False 

        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y):
                    found_bomb = True
                    bomb_dist = abs(self.x - x) + abs(self.y - y)  
                    min_dist = min(min_dist, bomb_dist)  
        if not found_bomb:
            return 100

        return min_dist if min_dist != float('inf') else 100

    def state_features(self, wrld, action) -> dict:

        max_dist = max(wrld.width(), wrld.height())
        features = {
            "bias": 1.0,
            "distance_to_monster": self.get_dist_to_monster(wrld) / max_dist,  
            "distance_to_exit": self.get_dist_to_exit(wrld) / max_dist,  # You may wish to adjust or rename this feature
            "bomb_risk": 1.0 if wrld.explosion_at(self.x, self.y) or wrld.bomb_at(self.x, self.y) else 0.0,
            "action_x": action[0] / max_dist,
            "action_y": action[1] / max_dist
        }
        
        # Use the section goal for computing the A* path
        goal = self.get_section_goal(wrld)
        a_star_path = self.astar(wrld, (self.x, self.y), goal, explosion_cells={}, current_time=wrld.time)
        if a_star_path:
            # Compute the minimal Manhattan distance from the current position to any point on the A* path.
            distance_to_path = min(abs(self.x - pos[0]) + abs(self.y - pos[1]) for pos in a_star_path)
        else:
            distance_to_path = 1
        features["a_star_alignment"] = -distance_to_path
        return features
    
    def get_q_value(self, wrld, action):
        action_key = f'action_{action}'
        #return self.weights.get(action_key, 0.0) 
        features = self.state_features(wrld, action)
        #return sum(self.weights[f] * features[f] for f in features)
        return sum(self.weights.get(f, 0.0) * features[f] for f in features)
    

    def get_best_action(self, wrld):
        actions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        goal = self.get_section_goal(wrld)
        astar_path = self.astar(wrld, (self.x, self.y), goal, {}, wrld.time)

        if np.random.rand() < self.epsilon:
            return actions[randint(0, len(actions) - 1)]
        #Use the A* suggestion if available
        if astar_path and len(astar_path) > 1:
            next_x, next_y = astar_path[1]
            best_action = (next_x - self.x, next_y - self.y)
            print(f"Following A* action toward section goal: {best_action}")
        #     return best_action
        else:
        # Fall back to Q-learning decision if no valid A* path is found.
            q_values = [(action, self.get_q_value(wrld, action)) for action in actions]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [action for action, q in q_values if q == max_q]
        self.update_epsilon()

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

        #adjusted_alpha = self.alpha * (1 - self.epsilon)
        adjusted_alpha = self.alpha

        # Update weights
        for f in features:
            if f not in self.weights:
                self.weights[f] = 0.0
            self.weights[f] += adjusted_alpha * error * features[f]
            print(f"Updated weights: {self.weights}")

    def actions_near_wall(self, wrld):
       place_bomb = False
       wall_action = np.array([place_bomb], dtype=bool)

       if self.bomb_cooldown > 0:
            self.bomb_cooldown -= 1
            #print(f"Bomb cooldown: {self.bomb_cooldown}")
            return wall_action

       wall_count = 0
       adjacent_walls = []
       for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                if wrld.wall_at(nx, ny):
                    wall_count += 1
                    adjacent_walls.append((nx, ny))  

       exit_x, exit_y = wrld.exitcell

       print(f"Wall Count: {wall_count}, Adjacent Walls: {adjacent_walls}")

    # If near at least one wall but not fully trapped
       if 1 <= wall_count <= 2:
            place_bomb = True
            self.placing_bomb = True
            print("Placing bomb near wall...")
            self.place_bomb()
            self.bomb_cooldown = 5

            print("Placed bomb near wall, now evading explosion!")
            self.evade_bomb(wrld, exit_x, exit_y)

       return wall_action
       #best_move = self.get_best_action(wrld)

       #print(f"Wall Count: {wall_count}, Best Move: {best_move}")

    #    if 1 <= wall_count <= 2:  # If there's at least one but not too many walls around
    #         new_x, new_y = self.x + best_move[0], self.y + best_move[1]

    #     # Ensure coordinates are within bounds before checking for walls
    #         if 0 <= new_x < wrld.width() and 0 <= new_y < wrld.height():
    #             if wrld.wall_at(new_x, new_y):
    #                 place_bomb = True
    #                 self.placing_bomb = True  
    #                 print("Placing bomb near wall...")
    #                 self.place_bomb()

    #                 print("Placed bomb near wall, now evading explosion!")
    #                 self.evade_bomb(wrld, exit_x, exit_y)
    #             else:
    #                 print(f"No wall at target ({new_x}, {new_y}), skipping bomb placement.")
    #         else:
    #             print(f"New position ({new_x}, {new_y}) is out of bounds, cannot place bomb.")
    
    #    return wall_action


    def get_reward(self, wrld, wall_action):
        reward = 0
        exit_x, exit_y = wrld.exitcell

        # Previous distance calculation
        if hasattr(self, 'prev_position'):
            prev_x, prev_y = self.prev_position
            prev_dist = abs(prev_x - exit_x) + abs(prev_y - exit_y)
        else:
            prev_dist = abs(self.x - exit_x) + abs(self.y - exit_y)
        new_dist = abs(self.x - exit_x) + abs(self.y - exit_y)
        self.prev_position = (self.x, self.y)
        
        if new_dist < prev_dist:
            reward += 20  
            print("Moved closer to exit! Reward: +20")
        
        bomb_distance = self.get_dist_to_bomb(wrld)
        if bomb_distance > 2:
            reward += 10
            print("Evaded bomb! Reward: +10")
        elif bomb_distance > 2:
            reward += 10
        
        if wall_action[0]:
            reward += 20  
            print("Smart bomb placement! Reward: +20")
        
        if wrld.explosion_at(self.x, self.y):
            reward -= 100  
            print("Hit by explosion! Penalty: -100")
        
        if any(m.x == self.x and m.y == self.y for sublist in wrld.monsters.values() for m in sublist):
            reward -= 50  
            print("Caught by monster! Penalty: -50")
        
        if hasattr(self, 'prev_position') and self.prev_position == (self.x, self.y):
            reward -= 10  
            print("Stood still! Penalty: -5")
        
        if bomb_distance <= 2:
            penalty = (3 - bomb_distance) * 5  
            reward -= penalty
            print(f"Too close to bomb! Penalty: -{penalty}")
        
        # *** A* Path Reward Shaping ***
        a_star_path = self.astar(wrld, (self.x, self.y), (exit_x, exit_y), explosion_cells={}, current_time=wrld.time)
        # If a valid A* path exists and the current position is on it, give a bonus.
        if a_star_path and (self.x, self.y) in a_star_path[:2]:
            reward += 40  
            print("On A* path! Bonus reward: +40")
        
        print(f"Final Reward: {reward}")
        return reward

