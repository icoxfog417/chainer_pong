import gym
from pong_dqn import Q, Agent, Trainer


def main():
    q = Q(200)
    agent = Agent(epsilon=0)  # totally greedy
    t = Trainer(gamma=0.99,
                batch_size=10,
                learning_rate=1e-4,
                decay_rate=0.99)
    
    env = gym.make("Pong-v0")
    t.train(q, agent, env)


if __name__ == "__main__":
    main()
 