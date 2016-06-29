import os
import sys
import gym
import numpy as np
import chainer
from chainer import serializers
from pong_dqn import Q, Agent, Trainer


def main(file_name, game_count=20):
    path = Trainer.model_path(file_name)
    if not os.path.isfile(file_name):
        print("{0} is not exist.".format(file_name))
    
    env = gym.make("Pong-v0")
    q = Q(200, env.action_space.n)
    agent = Agent(0.1, list(range(env.action_space.n)))
    serializers.load_npz(path, q)
    
    _prefix, _ext = os.path.splitext(path)
    for i_episode in range(game_count):
        observation = env.reset()
        continue_game = True
        prev = None
        while continue_game:
            env.render()
            s, a, _, _ = Trainer.act(observation, q, agent, prev)
            prev = s
            observation, reward, done, info = env.step(a)
            continue_game = not done


if __name__ == "__main__":
    argvs = sys.argv
    file_name = ""
    if len(argvs) == 2:
        file_name = argvs[1]
    main(file_name)
