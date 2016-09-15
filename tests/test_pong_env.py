import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model.pong_env import PongEnv
from model.rule_agent import RandomAgent


class TestPongEnv(unittest.TestCase):

    def test_run_environment(self):
        env = PongEnv()
        agent = RandomAgent(env.actions)
        env.play(agent, episode=1)


if __name__ == "__main__":
    unittest.main()


