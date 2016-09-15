import os
import sys
import gym


class PongEnv():

    def __init__(self):
        self.env = gym.make("Pong-v0")
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
                    action = agent.act(observation, reward, done, i)
                    observation, reward, done, info = self.env.step(action)
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
    
    @classmethod
    def format_observation(cls, observation):
        pass


class Agent():

    def __init__(self, actions):
        self.actions = actions

    def start(self, observation):
        return 0  # default action

    def act(self, observation, reward, done=False, episode_count=-1):
        return 0  # default action
    
    def report(self, scores):
        return ""

