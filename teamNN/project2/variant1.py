# This is necessary to find the main code
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
reward_history = []
for i in range(300):
    sys.path.insert(1, '../teamNN')
    from testcharacter import TestCharacter


    # Create the game
    g = Game.fromfile('map.txt')

    # TODO Add your character
    test_character = TestCharacter("me", # name
                                "C",  # avatar
                                 0, 0)  # position
    g.add_character(test_character)
    #reward_history.append(sum(reward for (_, _, reward, _, _) in list(test_character.memory)[-30:]))

    print("Iteration: ", i)
    g.go(1)

# plt.plot(reward_history)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Reward Progression Over Episodes")
# plt.show()

# Run!
#g.go()
