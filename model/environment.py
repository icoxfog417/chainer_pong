import os
import sys
import gym
from model.agent import Agent


class Environment():

    def __init__(self, env_name="Pong-v0"):
        self.env = gym.make(env_name)
        self.actions = list(range(self.env.action_space.n))
    
    def play(self, agent, episode=5, render=True, report_interval=-1):
        scores = []
        for i in range(episode):
            observation = self.env.reset()
            done = False
            reward = 0.0
            step_count = 0
            score = 0.0
            continue_game = True
            while continue_game:
                if render:
                    self.env.render()
                if step_count == 0:
                    action = agent.start(observation)
                else:
                    action = agent.act(observation, reward)
                    observation, reward, done, info = self.env.step(action)

                if done:
                    agent.end(observation, reward)
                
                yield i, step_count, reward

                continue_game = not done
                score += reward
                step_count += 1

            scores.append(score)

            if report_interval > 0 and i % report_interval == 0:
                print("average score is {0}.".format(sum(scores) / len(scores)))
                report = agent.report()
                if report:
                    print(report)
                scores = []
