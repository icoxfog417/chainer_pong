import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
import gym
import os
from collections import namedtuple


class Q(chainer.Chain):
    """
    You want to optimize this function to determine the action from state (state is represented by CNN vector)
    """

    D = 80 * 80  # input is 80 x 80 size grayed image
    
    def __init__(self, hidden):
        self.hidden = hidden
        super(Q, self).__init__(
            l1=L.Linear(self.D, hidden, wscale=1/np.sqrt(self.D)),
            l2=L.Linear(hidden, 1, wscale=1/np.sqrt(hidden))
        )
    
    def clear(self):
        self.loss = None
    
    def forward(self, x):
        h = F.relu(self.l1(x))
        p = F.sigmoid(self.l2(h))
        return p  # probability of taking UP action
        
    def __call__(self, x, t):
        self.clear()
        p = self.forward(x)
        td = t - p
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # avoid 0 division
        td_clip = td * (abs(td.data) <= 1) + td / abs(td_tmp) * (abs(td.data) > 1)
        zeros = chainer.Variable(np.zeros(td.data.shape, dtype=np.float32))
        self.loss = F.mean_squared_error(td_clip, zeros)
        return self.loss
        
       
class Agent():
    
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.actions = [2, 3]  # UP/DOWN
    
    def action(self, prob, force_greedy=False):
        """ This is Agent's policy """
        if np.random.rand() < self.epsilon and not force_greedy:
            action = self.actions(np.random.randint(0, len(self.actions)))
        else:
            action = 2 if np.random.uniform() < prob else 3 # roll the dice!
        return action
        

class Trainer():
    Episode = namedtuple("Episode", ["states", "targets"])
    
    def __init__(self, 
                    gamma=0.99, 
                    batch_size=10, 
                    learning_rate=1e-4, 
                    decay_rate=0.99):
        self.gamma = gamma  # discount factor for reward
        self.batch_size = batch_size  # memory size for experience replay
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.environment = gym.make("Pong-v0")
        self.optimizer = optimizers.RMSprop(lr=self.learning_rate, alpha=decay_rate)
        self.model_path = lambda fn: os.path.join(os.path.dirname(__file__), "models/" + fn)
    
    def _adjust(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float32).ravel()
 
    def _calculate_targets(self, teacher, agent, rewards, nexts):
        """ calculate teacher value """
        
        targets = np.zeros(len(rewards))
        length = len(rewards) - 1
        index = length
        for r in reversed(rewards):
            if index == length:
                t = np.sign(r)
            else:
                n = chainer.Variable(np.array([nexts[index]]))
                p = teacher.forward(n).data.flatten()[0]
                a = agent.action(p, force_greedy=True)
                q = 1 if a == 2 else 0  # teacher sign is 0/1
                t = np.sign(r) + self.gamma * q

            targets[index] = t
            index -= 1
        
        return targets
    
    def train(self, q_model, agent, env):
        """
        q model is optimized in accordance with openai gym environment. each action is decided by given agent
        """
        
        # setting up environment
        observation = env.reset()
        prev = None
        teacher = Q(q_model.hidden)
        self.optimizer.setup(q_model)
        
        # memory
        ss, ns, rs = [], [], []  # state, action, next state, reward
        episode_count = 0
        memory = []
        total_reward = 0
        running_reward = None

        while True:
            make_state = lambda i: self._adjust(i) - prev if prev is not None and i is not None else np.zeros(Q.D, dtype=np.float32)
            s = make_state(observation)
            prev = s
            up_prob = q_model.forward(chainer.Variable(np.array([s])))
            _up_prob = up_prob.data.flatten()[0]
            action = agent.action(_up_prob)
            
            # execute action and get new observation
            observation, reward, done, info = env.step(action)
            
            # momory it
            ss.append(s)
            ns.append(make_state(observation))
            rs.append(reward)
            total_reward += reward
            
            if done:
                print("episode {0} has done. its length is {1}.".format(episode_count, len(rs)))
                episode_count += 1
                tgts = self._calculate_targets(teacher, agent, rs, ns)
                ep = Trainer.Episode(np.array(ss, dtype=np.float32), np.array(tgts, dtype=np.float32))
                memory.append(ep)
                ss, ns, rs = [], [], []
                
                if episode_count % self.batch_size == 0:
                    self.optimizer.zero_grads()
                    for ep in memory:
                        indices = np.random.permutation(len(ep.states))  # random sampling
                        v_states = chainer.Variable(np.array(ep.states)[indices])
                        v_targets = chainer.Variable(np.vstack(ep.targets)[indices])
                        self.optimizer.update(q_model, v_states, v_targets)
                    
                    print("> done batch update. loss={0}".format(q_model.loss.data))
                    teacher.copyparams(q_model)  # update teacher

                running_reward = total_reward if running_reward is None else running_reward * self.gamma + total_reward * (1 - self.gamma)
                print("resetting env. episode reward total was {0}. running mean: {1}".format(total_reward, running_reward))
                if episode_count % 100 == 0:
                    serializers.save_npz(self.model_path("pong.model"), q_model)
                total_reward = 0
                observation = env.reset() # reset env
                prev = None
