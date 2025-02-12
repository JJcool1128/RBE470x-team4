import sys
sys.path.insert(0, '../bomberman')
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from math import sqrt
from game import Game

class TestCharacter(CharacterEntity):
    print("TestCharacter class loaded!")

    def do(self, wrld):
        print("Do() called!")
        print(wrld)
        start = (self.x, self.y)
        goal = (wrld.exitcell[0], wrld.exitcell[1])

        print(f"Start: {start}, Goal: {goal}")

        my_path = self.astar(wrld, start, goal, explosion_cells={}, current_time=wrld.time)
        print(f"Path: {my_path}")

        if not my_path or len(my_path) < 2:
            print("No valid path found or already at goal.")
            self.move(0, 0)  # Stay still if no valid path
            return

        next_step = my_path[1]
        dx, dy = next_step[0] - start[0], next_step[1] - start[1]

        print(f"Moving to: {next_step} with dx={dx}, dy={dy}")
        for monster in wrld.monsters.values():
            for m in monster:
                mx, my = m.x, m.y
                monster_distance = abs(self.x - mx) + abs(self.y - my)

                if monster_distance <= 3:
                    print("Monster detected! " , monster_distance, " away")
                    #self.place_bomb()
                    self.kill_monster(wrld)

        self.move(dx, dy)

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

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    
    def cost(self, wrld, a: tuple[int, int], b: tuple[int, int], explosion_cells, current_time) -> int:
        ax, ay = a
        bx, by = b
        final_cost = sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        if (bx, by) in explosion_cells:
            explosion_time = explosion_cells[(bx, by)]
            time_to_explode = explosion_time - current_time
            
            if time_to_explode > 3:
                final_cost += 20
            elif 1 <= time_to_explode <= 3:
                final_cost += 1000
            else:
                return float('inf')

        if wrld.monsters_at(bx, by) is not None:
            return float('inf')  

        for monster in wrld.monsters.values():
            for m in monster:
                mx, my = m.x, m.y
                monster_range = getattr(m, 'rnge', 0)
                monster_distance = abs(bx - mx) + abs(by - my)

                if monster_distance == 0:
                    return float('inf')  
                elif monster_distance <= monster_range:
                    final_cost += 100  
                elif monster_distance == monster_range + 1:
                    final_cost += 50
                elif monster_distance == monster_range + 2:
                    final_cost += 25
                elif monster_distance == monster_range + 3:
                    final_cost += 10

                # monster_moves = [(mx + dx, my + dy) for dx in [-2, -1, 0, 1, 2] for dy in [-2, -1, 0, 1, 2]
                #                  if 0 <= mx + dx < wrld.width() and 0 <= my + dy < wrld.height()
                #                  and not wrld.wall_at(mx + dx, my + dy)]

                monster_moves = [(mx + dx, my + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                 if 0 <= mx + dx < wrld.width() and 0 <= my + dy < wrld.height()
                                 and not wrld.wall_at(mx + dx, my + dy)]

                if (bx, by) in monster_moves:
                    final_cost += 200
        return final_cost 

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

        safe_path = self.astar(wrld, (current_x, current_y), (wrld.exitcell[0], wrld.exitcell[1]), explosion_cells, wrld.time)

        if safe_path and len(safe_path) > 1:
            print("Escape path found, placing bomb and moving!")
            self.place_bomb()
            next_x, next_y = safe_path[1]
            dx, dy = next_x - current_x, next_y - current_y
            self.move(dx, dy)
        else:
            print("No escape path! Bomb placement canceled.")
            self.move(0, 0)