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

for i in range(10):
    sys.path.insert(1, '../teamNN')
    from testcharacter import TestCharacter

    # Create the game
    random.seed(rand.randint(1,500)) # TODO Change this if you want different random choices
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("stupid", # name
                                "S",      # avatar
                                3, 9      # position
    ))

    # TODO Add your character
    g.add_character(TestCharacter("me", # name
                                "C",  # avatar
                                0, 0  # position
    ))

    # Run!
    g.go(1)
    time.sleep(2)