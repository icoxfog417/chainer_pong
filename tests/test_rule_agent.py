import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model.pong_env import PongEnv
from model.rule_agent import CycleAgent


class TestPongEnv(unittest.TestCase):

    def xtest_cycle_agent(self):
        env = PongEnv()
        agent = CycleAgent(env.actions, keep_length=200)
        env.play(agent, episode=3)

    def test_funfun_defence(self):
        env = PongEnv()
        agent = CycleAgent((2, 3), keep_length=20)
        env.play(agent, episode=2)


if __name__ == "__main__":
    unittest.main()


