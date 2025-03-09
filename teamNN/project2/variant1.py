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
from monsters.selfpreserving_monster import SelfPreservingMonster

reward_history = []

wins = 0
max_iterations = 1000

for i in range(max_iterations):
    sys.path.insert(1, '../teamNN')
    from testcharacter import TestCharacter

    # Create the game
    random.seed(123) # TODO Change this if you want different random choices

    g = Game.fromfile('map.txt')

    # g.add_monster(StupidMonster("stupid", # name
    #                             "S",      # avatar
    #                             3, 9      # position
    # ))

    # g.add_monster(SelfPreservingMonster("selfpreserving", # name
    #                                     "S",              # avatar
    #                                     3, 9,             # position
    #                                     1                 # detection range
    # ))
    
    # g.add_monster(SelfPreservingMonster("aggressive", # name
    #                                     "A",          # avatar
    #                                     3, 13,        # position
    #                                     2             # detection range
    # ))

    test_character = TestCharacter("me", "C", 0, 0)

    # TODO Add your character
    g.add_character(test_character)

    recent_transitions = list(test_character.memory)[-100:]  # up to last 100
    total_reward = sum(t[2] for t in recent_transitions)     # t[2] is 'reward'
    reward_history.append(total_reward)

    # (NEW) Check if we had a "win" in these transitions
    # Because the last element of each transition is 'won'
    if any(t[5] for t in recent_transitions):
        wins += 1

    if i > 0:
        print(f"Iteration {i} - Episode reward: {reward_history[i]}")
    g.go(1)

print(f"Wins: {wins}/{max_iterations}")

# Plot the reward history
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Progression Over Episodes")
plt.show()