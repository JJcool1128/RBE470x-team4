# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from math import sqrt
from game import Game
from state_machine import GameState, State, ReachGoal, AvoidMonster, AvoidAgressiveMonsters, AvoidTwoMonsters, StateMachine

class TestCharacter(CharacterEntity):
    print("TestCharacter class loaded!")

    def do(self, wrld):
        print("Do() called!")
        print(wrld)
        start = (self.x, self.y)
        goal = (wrld.exitcell[0] , wrld.exitcell[1])

        print(f"Start: {start}, Goal: {goal}")

        my_path = self.astar(wrld, start, goal)
        print(f"Path: {my_path}")

        if not my_path or len(my_path) < 2:
            print("No valid path found or already at goal.")
            return None

        next_step = my_path[1]  # Get the next move in the path
        dx, dy = next_step[0] - start[0], next_step[1] - start[1]
    
        print(f"Moving to: {next_step} with dx={dx}, dy={dy}")

        if wrld.wall_at(next_step[0], next_step[1]):
            print(f"Wall in the way at {next_step}! Recalculating path...")
            my_path = self.astar(wrld, start, goal)
            if not my_path or len(my_path) < 2:
                next_step = my_path[1]
                dx, dy = next_step[0] - self.x, next_step[1] - self.y

        self.move(dx, dy)

        if (self.x, self.y) == goal:
            print("Reached the goal!")
             

    # NEIGHBORS OF 4 FOR VARIANT 1
    print("finding neighbors of 4")
    def neighbors_of_4(self, wrld, current: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = []
        a, b = current

        directions = [
            (0, 1),   # up
            (0, -1),  # down
            (1, 0),   # right
            (-1, 0)  # left
        ]
        #print(f"Current: {current}")
        for dx, dy in directions:
            nx = a + dx
            ny = b + dy
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                if not wrld.wall_at(nx, ny):
                    neighbors.append((nx, ny))

        #print(f"Neighbors: {neighbors}")
        return neighbors


    print("heuristic")
    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        ax = a[0]
        ay = a[1]
        bx = b[0]
        by = b[1]


        return abs(ax - bx) + abs(ay - by) # Manhattan distance
    
    print("cost")
    def cost(self, wrld, a: tuple[int, int], b: tuple[int, int]) -> int:
        ax = a[0]
        ay = a[1]
        bx = b[0]
        by = b[1]

        final_cost = sqrt(pow(ax-bx, 2) + pow(ay-by, 2)) # Euclidean distance

        # if wrld.monster_at(b[0], b[1]) is not None and len(wrld.monster_at(b[0], b[1])) > 0:
        #     return final_cost + 1

        return final_cost 

    print("astar")
    def astar(self, wrld, start: tuple[int,int], goal: tuple[int,int]) -> list[tuple[int,int]]:
        print("A* called!")
        frontier = PriorityQueue()
        #print(f"A* Start: {start} (Type: {type(start)}) | Goal: {goal} (Type: {type(goal)})")

        frontier.put((0, start))
        came_from = {}
        cost_so_far = {}
        cost_so_far[start] = 0
        came_from[start] = None

        while not frontier.empty():
            _, current = frontier.get()

            if current == goal:
                break

            for next in self.neighbors_of_4(wrld, current):
                new_cost = cost_so_far[current] + self.cost(wrld, current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put((priority, next))
                    came_from[next] = current
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()  # Start -> Goal order
        return path


    # def is_cell_walkable(self, wrld, a: tuple[int, int]) -> bool:
    #     ax = a[0]
    #     ay = a[1]

    #     map_boundary_x = wrld.width()
    #     map_boundary_y = wrld.height()
    #     cell_walkable = True

    #     if ax < 0 or ax >= map_boundary_x or ay < 0 or ay >= map_boundary_y:
    #         cell_walkable = False
    #     elif wrld.wall_at(ax, ay):
    #         cell_walkable = False
            
    #     elif wrld.monster_at(ax, ay) is not None and len(wrld.monster_at(ax, ay)) > 0:
    #         cell_walkable = False     

    #     return cell_walkable
        
    
    # def neighbors_of_8(self, wrld, current: tuple[int, int]) -> list[tuple[int, int]]:
    #     neighbors = []
    #     a, b = current

    #     directions = [
    #         (0, 1),   # up
    #         (0, -1),  # down
    #         (1, 0),   # right
    #         (-1, 0),  # left
    #         (1, 1),   # diagonal bottom-right
    #         (-1, -1), # diagonal top-left
    #         (1, -1),  # diagonal bottom-left
    #         (-1, 1)   # diagonal top-right
    #     ]
    #     #print(f"Current: {current}")
    #     for dx, dy in directions:
    #         nx, ny = a + dx, b + dy
    #         if self.is_cell_walkable(wrld, nx, ny):
    #             neighbors.append((nx, ny))

    #     return neighbors



    
        


        

