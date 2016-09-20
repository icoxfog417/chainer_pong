import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model.environment import Environment
from model.rule_agent import CycleAgent


class TestRuleAgent(unittest.TestCase):

    def test_cycle_agent(self):
        env = Environment()
        agent = CycleAgent(env.actions, keep_length=200)
        for episode, step, reward in env.play(agent, episode=3):
            pass

    def test_funfun_defence(self):
        env = Environment(env_name="Pong-v0")
        agent = CycleAgent((2, 3), keep_length=20)

        for episode, step, reward in env.play(agent, episode=1):
            pass


if __name__ == "__main__":
    unittest.main()


