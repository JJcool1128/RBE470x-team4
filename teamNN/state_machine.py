from enum import Enum

class GameState(Enum):
    REACH_GOAL = 1
    AVOID_MONSTER = 2
    AVOID_AGRESSIVE_MONSTERS = 3
    AVOID_TWO_MONSTERS = 4

class State:
    "Base class for all states"
    def execute(self, character, wrld):
        pass

class ReachGoal(State):
    "State where the character tries to reach the goal"
    def execute(self, character, wrld):
        self.state = GameState.REACH_GOAL

class AvoidMonster(State):
    "State where the character tries to avoid a monster"
    def execute(self, character, wrld):
        self.state = GameState.AVOID_MONSTER
    
    

class StateMachine():
    def __init__(self, initial_state: GameState):
        self.current_state = initial_state
        self.states = {GameState.REACH_GOAL: ReachGoal(),
                       GameState.AVOID_MONSTER: AvoidMonster()}





