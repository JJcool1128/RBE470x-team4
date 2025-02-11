# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from queue import PriorityQueue
from math import sqrt
from game import Game
#from state_machine import GameState, State, ReachGoal, AvoidMonster, AvoidSelfPreservMonster, AvoidAggressiveMonsters, AvoidTwoMonsters, StateMachine


 
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

        for monster in wrld.monsters.values():
            for m in monster:
                mx, my = m.x, m.y
                monster_distance = abs(self.x - mx) + abs(self.y - my)

                if monster_distance <= 3:
                    print("Monster detected! " , monster_distance, " away")
                    #self.place_bomb()
                    #self.kill_monster(wrld)

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

        if wrld.monsters_at(bx, by) is not None:
            return float('inf')  

        # for dx in [-1, 0, 1]:
        #     for dy in [-1, 0, 1]:
        #         nx, ny = bx + dx, by + dy
        #         if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
        #             if wrld.monsters_at(nx, ny) is not None:
        #                 final_cost += 10  

        for monster in wrld.monsters.values():
            for m in monster:
                mx, my = m.x, m.y
                monster_range = getattr(m, 'rnge', 0)
                monster_distance = abs(bx - mx) + abs(by - my)


                if monster_distance == 0:
                    return float('inf')  
                elif monster_distance <= monster_range:
                    final_cost += 50  
                elif monster_distance == monster_range + 1:
                    final_cost += 25
                elif monster_distance == monster_range + 2:
                    final_cost += 12
                elif monster_distance == monster_range + 3:
                    final_cost += 6

                monster_moves = [(mx + dx, my + dy) for dx in [-2 ,-1, 0, 1, 2] for dy in [-2 ,-1, 0, 1, 2]
                              if 0 <= mx + dx < wrld.width() and 0 <= my + dy < wrld.height()
                              and not wrld.wall_at(mx + dx, my + dy)]

                if (bx, by) in monster_moves:
                    final_cost += 100
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

    def kill_monster(self, wrld):
        current_x, current_y = self.x, self.y
        self.place_bomb()

        explosion_cells = {}  
        for bomb in wrld.bombs.values():
            if bomb.x == current_x and bomb.y == current_y:
                explosion_range = wrld.explosion_range  
                explosion_time = bomb.timer  
                explosion_duration = wrld.explosion_duration  

                
                for d in range(-explosion_range, explosion_range + 1):
                    if 0 <= current_x + d < wrld.width():
                        explosion_cells[(current_x + d, current_y)] = explosion_time + explosion_duration
                    if 0 <= current_y + d < wrld.height():
                        explosion_cells[(current_x, current_y + d)] = explosion_time + explosion_duration

        safe_path = self.astar(wrld, (current_x, current_y), (wrld.exitcell[0], wrld.exitcell[1]), explosion_cells, wrld.time)

        if safe_path and len(safe_path) > 1:
            next_x, next_y = safe_path[1]
            dx, dy = next_x - current_x, next_y - current_y
            self.move(dx, dy)
        else:
            print("No path found!")


    

            
        
        
            
        



    # def is_cell_walkable(self, wrld, a: tuple[int, int]) -> bool:
    #     ax = a[0]
    #     ay = a[1]

    #     map_boundary_x = wrld.width()
    #     map_boundary_y = wrld.height()
    #     cell_walkable = True

    # #     if ax < 0 or ax >= map_boundary_x or ay < 0 or ay >= map_boundary_y:
    # #         cell_walkable = False
    # #     elif wrld.wall_at(ax, ay):
    # #         cell_walkable = False
    # #     if ax < 0 or ax >= map_boundary_x or ay < 0 or ay >= map_boundary_y:
    # #         cell_walkable = False
    # #     elif wrld.wall_at(ax, ay):
    # #         cell_walkable = False
            
    # #     elif wrld.mons_at(ax, ay) is not None and len(wrld.mons_at(ax, ay)) > 0:
    # #         cell_walkable = False     

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



    
        


        

