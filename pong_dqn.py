import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
import os
from collections import namedtuple


class Q(chainer.Chain):
    """
    You want to optimize this function to determine the action from state (state is represented by CNN vector)
    """

    D = 80 * 80  # input is 80 x 80 size grayed image
    
    def __init__(self, hidden, action_count):
        self.hidden = hidden
        self.action_count = action_count
        super(Q, self).__init__(
            l1=L.Linear(self.D, hidden, wscale=np.sqrt(self.D)),
            l2=L.Linear(hidden, hidden, wscale=np.sqrt(hidden)),
            l3=L.Linear(hidden, action_count, wscale=np.sqrt(hidden))
        )
    
    def clone(self):
        return Q(self.hidden, self.action_count)
    
    def clear(self):
        self.loss = None
    
    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        qv = self.l3(h2)
        return qv
        
       
class Agent():
    
    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
    
    def action(self, qv, force_greedy=False):
        """ This is Agent's policy """
        is_greedy = True
        if np.random.rand() < self.epsilon and not force_greedy:
            action = self.actions[np.random.randint(0, len(self.actions))]
            is_greedy = False
        else:
            action = np.argmax(qv)
            print(qv)
        return action, is_greedy
        

class Trainer():
    Episode = namedtuple("Episode", ["states", "actions", "next_states", "rewards"])
    
    def __init__(self, 
                    gamma=0.99, 
                    batch_size=10, 
                    learning_rate=1e-4, 
                    decay_rate=0.99,
                    initial_epsilon=1,
                    epsilon_decay=0.99,
                    minimum_epsilon=0.1):
        self.gamma = gamma  # discount factor for reward
        self.batch_size = batch_size  # memory size for experience replay
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.optimizer = optimizers.RMSprop(lr=self.learning_rate, alpha=decay_rate)
    
    @classmethod
    def model_path(cls, fn="pong.model"):
        _fn = fn if fn else "pong.model"
        return os.path.join(os.path.dirname(__file__), "models/" + _fn)
    
    @classmethod
    def act(cls, observation, q_model, agent, prev=None):
        s, merged = cls._make_input(observation, prev)
        qv = q_model.forward(chainer.Variable(np.array([merged])))
        action, is_greedy = agent.action(qv.data.flatten())
        return s, action, is_greedy, np.max(qv.data)
    
    @classmethod
    def _make_input(cls, observation, prev):
        s = cls._adjust(observation) if observation is not None else np.zeros(Q.D, dtype=np.float32)
        merged = np.maximum(s, prev) if prev is not None else np.zeros(Q.D, dtype=np.float32)
        return s, merged

    @classmethod
    def _adjust(cls, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float32).ravel()

    def calculate_loss(self, q_model, teacher, episode, indices=()):
        shuffle = lambda x: x if len(indices) == 0 else x[indices]
        states, actions, next_states, rewards = \
            [shuffle(d) for d in (episode.states, episode.actions, episode.next_states, episode.rewards)]
        
        v_states = chainer.Variable(states)
        v_next_states = chainer.Variable(next_states)

        qv = q_model.forward(v_states)
        max_qv_next = np.max(teacher.forward(v_next_states).data, axis=1)
        teacher_qv = np.sign(rewards) + self.gamma * max_qv_next
        target = qv.data.copy()
        
        for i, a in enumerate(actions):
            target[i, a] = teacher_qv[i]

        td = chainer.Variable(target) - qv
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # avoid 0 division
        td_clip = td * (abs(td.data) <= 1) + td / abs(td_tmp) * (abs(td.data) > 1)
        zeros = chainer.Variable(np.zeros(td.data.shape, dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zeros)
        return loss
        
     
    def train(self, q_model, env, render=False):
        """
        q model is optimized in accordance with openai gym environment. each action is decided by given agent
        """
        
        # setting up environment
        observation = env.reset()
        prev = None
        self.optimizer.setup(q_model)
        agent = Agent(self.initial_epsilon, list(range(env.action_space.n)))
        teacher = q_model.clone()
        
        # memory
        ss, acs, ns, rs = [], [], [], []  # state, action, next state, reward
        episode_count = 0
        memory = []
        total_reward = 0
        running_reward = None

        while True:
            if render: env.render()
            s, a, is_g, q_max = self.act(observation, q_model, agent, prev)
            prev = s
            
            # execute action and get new observation
            observation, reward, done, info = env.step(a)
            print("action={0} by {1}. reward={2}".format(a, "greedy(qvalue={0})".format(q_max) if is_g else "random", reward))

            # momory it
            ss.append(s)
            acs.append(a)
            ns.append(self._make_input(observation, prev)[1])
            rs.append(reward)
            total_reward += reward
            
            if done:
                print("episode {0} has done. its length is {1}.".format(episode_count, len(rs)))
                episode_count += 1
                ep = Trainer.Episode(
                    np.array(ss, dtype=np.float32),
                    np.array(acs, dtype=np.int8),
                    np.array(ns, dtype=np.float32),
                    np.array(rs, dtype=np.float32))
                memory.append(ep)
                ss, acs, ns, rs = [], [], [], []
                
                if episode_count % self.batch_size == 0:
                    self.optimizer.zero_grads()
                    total_loss = 0
                    for ep in memory:
                        indices = np.random.permutation(len(ep.states))  # random sampling
                        loss = self.calculate_loss(q_model, teacher, ep, indices)
                        total_loss += loss.data
                        loss.backward()
                    self.optimizer.update()

                    print("> done batch update. loss={0}".format(total_loss))
                    teacher.copyparams(q_model)  # update teacher
                    memory = []
                
                # update policy
                agent.epsilon -= self.epsilon_decay
                if agent.epsilon < self.minimum_epsilon:
                    agent.epsilon = self.minimum_epsilon
                
                # logging
                running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
                print("resetting env. episode reward total was {0}. running mean: {1}, epsilon: {2}"
                    .format(total_reward, running_reward, agent.epsilon))
                if episode_count % 100 == 0:
                    serializers.save_npz(self.model_path(), q_model)
                total_reward = 0
                observation = env.reset() # reset env
                prev = None
