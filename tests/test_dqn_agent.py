import os
import sys
import shutil
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from PIL import Image
import numpy as np
from model.environment import Environment
from model.agent import Agent
from model.dqn_agent import DQNAgent, Q


class FormatAgent(Agent):

    def __init__(self, path):
        self.agent = DQNAgent((0, 1))
        self.path = path
        self._step = 0

    def act(self, observation, reward):
        arr = self.agent._format(observation)
        img = Image.fromarray((arr * 255).astype(np.uint8))  # because 0/1 value
        img.save(os.path.join(self.path, "image_{0}.png".format(self._step)), "png")
        self._step += 1
        return 0


class TestDQNAgent(unittest.TestCase):
    IMG_PATH = ""

    @classmethod
    def setUpClass(cls):
        cls.IMG_PATH = os.path.join(os.path.dirname(__file__), "./images")
        if not os.path.exists(cls.IMG_PATH):
            os.mkdir(cls.IMG_PATH)
    
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.IMG_PATH)

    def test_format_image(self):
        agent = FormatAgent(self.IMG_PATH)
        env = Environment()
        for ep, s, r in env.play(agent, episode=1):
            pass
        img = Image.open(os.path.join(self.IMG_PATH, "image_0.png"))
        self.assertTrue(img)
        arr = np.asarray(img)
        self.assertTrue(arr.shape, (Q.SIZE, Q.SIZE))
    
    def test_save_state(self):
        env = Environment()
        agent = DQNAgent(env.actions)

        zeros = np.zeros((agent.q.SIZE, agent.q.SIZE), np.float32)
        pre_state = None
        for ep, s, r in env.play(agent, episode=1):
            state = agent.get_state()
            self.assertEqual(agent.q.n_history, len(state))
            last_state = np.maximum(agent._observations[0], agent._observations[-1])

            if s == 0:
                # after first action
                self.assertEqual(0, np.sum(zeros != agent._observations[-1]))
                self.assertEqual(1, len(agent._state))

            if s < agent.q.n_history:
                # until n_history
                self.assertEqual(0, np.sum(last_state != state[s]))
                if pre_state is not None:
                    self.assertEqual(0, np.sum(pre_state != state[s - 1]))
            else:
                # over n_history
                self.assertEqual(0, np.sum(last_state != state[-1]))
                if pre_state is not None:
                    self.assertEqual(0, np.sum(pre_state != state[-2]))

            pre_state = last_state.copy()



if __name__ == "__main__":
    unittest.main()


