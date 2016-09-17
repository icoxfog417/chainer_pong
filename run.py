import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from model.environment import Environment
from model.dqn_agent import DQNAgent
from model.dqn_trainer import DQNTrainer


PATH = os.path.join(os.path.dirname(__file__), "./store")


def run(render):
    env = Environment()
    agent = DQNAgent(env.actions, epsilon=0.1, model_path=PATH)

    for ep, s, r in env.play(agent, render=render):
        pass


def train(render):
    env = Environment()
    agent = DQNAgent(env.actions, epsilon=1, model_path=PATH)
    trainer = DQNTrainer(agent)

    for ep, s, r in env.play(trainer, episode=10**5, render=render, report_interval=10):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong DQN")
    parser.add_argument("--render", action="store_const", const=True, default=True, help="render or not")
    parser.add_argument("--train", action="store_const", const=True, default=False, help="train or not")
    args = parser.parse_args()

    if args.train:
        train(args.render)
    else:
        run(args.render)

