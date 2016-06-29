import argparse
import gym
from pong_dqn import Q, Trainer


def main(render=False):
    env = gym.make("Pong-v0")
    q = Q(200, env.action_space.n)
    t = Trainer(gamma=0.99,
                batch_size=10,
                learning_rate=1e-4,
                decay_rate=0.99,
                initial_epsilon=0.8,
                epsilon_decay=1.0/10**4,
                minimum_epsilon=0.1)
    
    t.train(q, env, render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pong DQN")
    parser.add_argument("--render", action="store_const", const=True, default=False, help="render or not")
    args = parser.parse_args()

    main(args.render)
