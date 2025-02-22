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
        self.active_bomb = False
        self.explosion_duration = 0

    # Main decision method called each turn
    def do(self, wrld):
        print("Do() called!")

        start = (self.x, self.y)
        goal = (wrld.exitcell[0], wrld.exitcell[1])

        if self.active_bomb:
            if self.explosion_duration == 4:
                self.move(-1, 0)
                self.explosion_duration -= 1
                return
            if self.explosion_duration == 3:
                self.move(0, -1)
                self.explosion_duration -= 1
                return
            if self.explosion_duration == 2 or self.explosion_duration == 1:
                self.move(0, 0)
                self.explosion_duration -= 1
                return
            else:
                self.explosion_duration = 0
                self.active_bomb = False 
                return
        # Check for nearby monsters; if detected, use minimax escape strategy.
        # for monster in wrld.monsters.values():
        #     for m in monster:
        #         mx, my = m.x, m.y
        #         monster_distance = float(sqrt(abs(self.x - mx)**2 + abs(self.y - my)**2))
        #         if monster_distance < 4:
        #             print(f"Monster detected {monster_distance} tiles away! Switching to minimax escape!")
        #             self.escaping = True
        #             self.escape_monster(wrld)
        #             return  

        self.escaping = False
        
        a, b = self.x, self.y
        if a+1== wrld.width() and wrld.wall_at(a, b+1) and not self.escaping:
            self.place_bomb()
            self.explosion_duration = 4
            self.active_bomb = True

        # Calculate path to goal using A*
        my_path = self.astar(wrld, start, goal)
        print(f"Path: {my_path}")

        if not my_path or len(my_path) < 2:
            print("No valid path found or already at goal.")
            self.move(0, 0)
            return

        next_step = my_path[1]
        dx, dy = next_step[0] - start[0], next_step[1] - start[1]
        print(f"Moving to: {next_step} with dx={dx}, dy={dy}")
        self.move(dx, dy)
    
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
    def cost(self, wrld, a: tuple[int, int], b: tuple[int, int]) -> int:
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

    # A* algorithm to find a path to the goal, avoiding hazards
    def astar(self, wrld, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        print("A* started...")
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
    

        closest_to_wall = start 
        closest_to_wall_dist = self.heuristic(start, goal)

        while not frontier.empty():
            _, current = frontier.get()
            
            dist_to_goal = self.heuristic(current, goal)
            if dist_to_goal < closest_to_wall_dist:
                closest_to_wall_dist = dist_to_goal
                closest_to_wall = current

            if current == goal:
                print("Goal reached!")
                break
            
            for next in self.neighbors_of_4(wrld, current):
                new_cost = cost_so_far[current] + self.cost(wrld, current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put((priority, next))
                    came_from[next] = current

        last_node = goal if goal in came_from else closest_to_wall
        # if goal not in came_from:
        #     print("No valid path to goal!")
        #     return []

        path = []
        current = last_node
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        print(f"Final path: {path}")
        return path


