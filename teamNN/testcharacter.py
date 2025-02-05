# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from math import sqrt

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        start = (self.x, self.y)
        goal = (wrld.exitcell[0], wrld.exitcell[1])

        my_path = self.astar(wrld, start, goal)
        if len(my_path) > 1:
            x2, y2 = my_path[1]
            self.move(x2 - start[0], y2 - start[1])

    def find_neighbors(self, wrld, current: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = []
        a, b = current

        directions = [
            (0, 1)
            (0, -1)
            (1, 0)
            (-1, 0)
        ]

        for dx, dy in directions:
            nx = a + dx
            ny = b + dy
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny):
                neighbors.append((nx, ny))

        return neighbors


    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        ax = a[0]
        ay = a[1]
        bx = b[0]
        by = b[1]


        return abs(ax - bx) + abs(ay - by) # Manhattan distance
    
    def cost(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        ax = a[0]
        ay = a[1]
        bx = b[0]
        by = b[1]

        final_cost = sqrt(pow(ax-bx, 2) + pow(ay-by, 2)) # Euclidean distance
        return final_cost 

    def astar(self, wlrd, start: tuple[int,int], goal: tuple[int,int]) -> list[tuple[int,int]]:
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        cost_so_far[start] = 0
        came_from[start] = None

        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for next in wlrd.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
            path.reverse()  # Start -> Goal order
        return path

            


    
        

