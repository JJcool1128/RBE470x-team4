import sys
sys.path.insert(0, '../bomberman')
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from math import sqrt
from game import Game

class TestCharacter(CharacterEntity):
    print("TestCharacter class loaded!")

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.escaping = False  

    # Main decision method called each turn
    def do(self, wrld):
        print("Do() called!")
        start = (self.x, self.y)
        goal = (wrld.exitcell[0], wrld.exitcell[1])

        # Check for nearby monsters; if detected, use minimax escape strategy.
        for monster in wrld.monsters.values():
            for m in monster:
                mx, my = m.x, m.y
                monster_distance = float(sqrt(abs(self.x - mx)**2 + abs(self.y - my)**2))
                if monster_distance < 4:
                    print(f"Monster detected {monster_distance} tiles away! Switching to minimax escape!")
                    self.escaping = True
                    self.escape_monster(wrld)
                    return  

        self.escaping = False

        # Calculate path to goal using A*
        my_path = self.astar(wrld, start, goal, explosion_cells={}, current_time=wrld.time)
        print(f"Path: {my_path}")

        if not my_path or len(my_path) < 2:
            print("No valid path found or already at goal.")
            self.move(0, 0)
            return

        next_step = my_path[1]
        dx, dy = next_step[0] - start[0], next_step[1] - start[1]
        print(f"Moving to: {next_step} with dx={dx}, dy={dy}")
        self.move(dx, dy)

    # Determine game state based on monster proximity
    def determine_state(self, wrld):
        monsters = [(m.x, m.y) for sublist in wrld.monsters.values() for m in sublist]
        distances = [abs(self.x - mx) + abs(self.y - my) for mx, my in monsters]
        if not distances:
            return GameState.SAFE

        closest = min(distances)
        if closest == 1:
            return GameState.IN_FRONT_OF_MONSTER
        elif 2 <= closest <= 4:  
            return GameState.NEAR_MONSTER
        elif len(distances) >= 2 and all(d <= 5 for d in distances):
            return GameState.IN_BETWEEN_TWO_MONSTERS
        else:
            return GameState.SAFE  

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

    # A* algorithm to find a path to the goal, avoiding hazards
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

    # Attempt to kill a monster by placing a bomb and escaping
    def kill_monster(self, wrld):
        current_x, current_y = self.x, self.y
        explosion_cells = {}
        explosion_range = wrld.expl_range  
        explosion_time = wrld.time + 1  
        explosion_duration = wrld.expl_duration  
        for d in range(-explosion_range, explosion_range + 1):
            if 0 <= current_x + d < wrld.width():
                explosion_cells[(current_x + d, current_y)] = explosion_time + explosion_duration
            if 0 <= current_y + d < wrld.height():
                explosion_cells[(current_x, current_y + d)] = explosion_time + explosion_duration
        safe_moves = []
        for move in self.neighbors_of_4(wrld, (current_x, current_y)):
            if move not in explosion_cells:
                safe_moves.append(move)
        if safe_moves:
            best_move = max(safe_moves, key=lambda m: self.heuristic(m, (wrld.exitcell[0], wrld.exitcell[1])))
            dx, dy = best_move[0] - current_x, best_move[1] - current_y
            print(f" Fast escape to {best_move}, avoiding explosion!")
            self.place_bomb()
            self.move(dx, dy)
            return 
        print(" No immediate safe move. Using A* for escape.")
        safe_path = self.astar(wrld, (current_x, current_y), (wrld.exitcell[0], wrld.exitcell[1]), explosion_cells, wrld.time)
        if safe_path and len(safe_path) > 2:
            print(" Found a longer escape path!")
            self.place_bomb()
            next_x, next_y = safe_path[1]
            dx, dy = next_x - current_x, next_y - current_y
            self.move(dx, dy)
        else:
            print(" No safe escape! Bomb placement canceled.")
