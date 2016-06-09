import gym
from pong_dqn import Q, Agent, Trainer


def main():
    env = gym.make("Pong-v0")
    q = Q(200, env.action_space.n)
    t = Trainer(gamma=0.99,
                batch_size=8,
                learning_rate=1e-4,
                decay_rate=0.99,
                initial_epsilon=1,
                epsilon_decay=1.0/10**6,
                minimum_epsilon=0.1)
    
    t.train(q, env, render=False)

if __name__ == "__main__":
    main()
 