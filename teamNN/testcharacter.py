# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')

# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import heapq

class TestCharacter(CharacterEntity):
    def do(self, wrld):
        """Moves the character towards the exit using A*."""
        next_move = navigate_character(wrld, self)
        if next_move:
            dx, dy = next_move[0] - self.x, next_move[1] - self.y
            self.move(dx, dy)

# A* Pathfinding Implementation
def a_star_search(wrld, start, goal):
    """Finds the shortest path from start to goal using A*, avoiding dangers."""
    def heuristic(a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        x, y = current
        neighbors = [(x+dx, y+dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]]

        for next_pos in neighbors:
            nx, ny = next_pos

            # Check if within bounds and avoid obstacles
            if (0 <= nx < wrld.width() and 0 <= ny < wrld.height() and 
                not wrld.wall_at(nx, ny) and
                not wrld.bomb_at(nx, ny) and
                not wrld.explosion_at(nx, ny) and
                not wrld.monsters_at(nx, ny)):

                new_cost = cost_so_far[current] + 1  # Uniform movement cost

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()

    return path if path and path[0] == start else None  # Ensure valid path

# Function to find the character's position and exit
def find_character_and_exit(wrld, character):
    """Finds the starting position of the character and the exit."""
    start, goal = None, None

    for x in range(wrld.width()):
        for y in range(wrld.height()):
            if wrld.me(character) and (x, y) == (wrld.me(character).x, wrld.me(character).y):
                start = (x, y)
            if wrld.exit_at(x, y):
                goal = (x, y)

    return start, goal

# Character navigation using A*
def navigate_character(wrld, character):
    """Moves the character towards the exit using A*, avoiding dangers."""
    start, goal = find_character_and_exit(wrld, character)
    if start and goal:
        path = a_star_search(wrld, start, goal)
        if path and len(path) > 1:
            return path[1]  # Move to the next step in the path
    return None  # No valid move found
