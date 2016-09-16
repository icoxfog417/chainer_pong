import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model.environment import Environment
from model.dqn_agent import DQNAgent
from model.dqn_trainer import DQNTrainer


class TestDQNTrainer(unittest.TestCase):

    def test_trainer(self):
        env = Environment()
        agent = DQNAgent(env.actions, epsilon=1)
        trainer = DQNTrainer(
            agent, 
            memory_size=10, 
            replay_size=5, 
            initial_exploration=8)
        
        global_step = 0
        last_state = []
        for ep, s, r in env.play(trainer, episode=2):
            if global_step < trainer.initial_exploration:
                self.assertEqual(1, trainer.agent.epsilon)
            else:
                self.assertTrue(trainer.agent.epsilon < 1)
                
            if len(last_state) != 0:
                _index = global_step % trainer.memorize
                

            last_state = agent.get_state()
            global_step += 1


if __name__ == "__main__":
    unittest.main()


