import random
from model.agent import Agent


class RandomAgent(Agent):
    """
    Most simple agent that acts at random.
    """

    def __init__(self, actions):
        self.actions = actions
        self._random_act = lambda: random.sample(self.actions, 1)[0]

    def start(self, observation):
        return self._random_act()

    def act(self, observation, reward):
        a = self._random_act()
        return self._random_act()


class CycleAgent(Agent):
    """
    Walk around given actions. You can use this agent to confirm action behavior
    """

    def __init__(self, actions, keep_length=10, initial_index=0):
        self.actions = actions
        self.keep_length = keep_length
        self._initial_index = initial_index
        self._index = initial_index
        if self._index >= len(actions):
            raise Exception("Initial Index is too large. max is {0} but {1}.".format(len(self.actions), initial_index))
        self._count = 0

    def start(self, observation):
        self._index = self._initial_index
        self._count = 0
        return self.actions[self._index]

    def act(self, observation, reward):
        action = self.actions[self._index]
        self._count += 1
        if self._count == self.keep_length:
            self._index = (self._index + 1) % len(self.actions)
            self._count = 0
            print("change action to {0}".format(self.actions[self._index]))
        return action
    
    def report(self, episode):
        return "keeping action={0}".format(self.actions[self._index])
