# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
import time
import random as rand
import matplotlib.pyplot as plt

reward_history = []

for i in range(100):
    sys.path.insert(1, '../teamNN')
    from testcharacter import TestCharacter

    # Create the game
    random.seed(123) # TODO Change this if you want different random choices
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("stupid", # name
                                "S",      # avatar
                                3, 9      # position
    ))

    test_character = TestCharacter("me", "C", 0, 0)

    # TODO Add your character
    g.add_character(test_character)

    reward_history.append(sum(reward for (_, _, reward, _, _) in list(test_character.memory)[-100:]))

    print("Iteration: ", i)
    g.go(1)

plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Progression Over Episodes")
plt.show()