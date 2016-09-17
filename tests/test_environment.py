import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model.environment import Environment
from model.rule_agent import RandomAgent


class TestEnvironment(unittest.TestCase):

    def test_run_environment(self):
        env = Environment()
        agent = RandomAgent(env.actions)
        for episode, step, reward in env.play(agent, episode=1):
            pass


if __name__ == "__main__":
    unittest.main()


