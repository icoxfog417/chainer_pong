import os
import shutil
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model.environment import Environment
from model.dqn_agent import DQNAgent
from model.dqn_trainer import DQNTrainer


class TestDQNTrainer(unittest.TestCase):
    MODEL_PATH = ""

    @classmethod
    def setUpClass(cls):
        cls.MODEL_PATH = os.path.join(os.path.dirname(__file__), "./store")
    
    @classmethod
    def tearDownClass(cls):
        #shutil.rmtree(cls.MODEL_PATH)
        pass

    def test_trainer(self):
        env = Environment()
        agent = DQNAgent(env.actions, epsilon=1, model_path=self.MODEL_PATH)
        trainer = DQNTrainer(
            agent, 
            memory_size=100, 
            replay_size=10, 
            initial_exploration=2000,
            target_update_freq=100,
            epsilon_decay=0.1
            )
        
        global_step = -1  # because "step" of trainer is count of train, so first start is not counted
        last_state = []
        for ep, s, r in env.play(trainer, episode=2, report_interval=1):
            if global_step < trainer.initial_exploration:
                self.assertEqual(1, trainer.agent.epsilon)
            else:
                self.assertTrue(trainer.agent.epsilon < 1)

            global_step += 1
            last_state = agent.get_state()


if __name__ == "__main__":
    unittest.main()


