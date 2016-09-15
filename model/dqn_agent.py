import numpy as np
from chainer import Variable
import chainer.functions as F


class Q(chainer.Chain):
    """
    You want to optimize this function to determine the action from state (state is represented by CNN vector)
    """

    SIZE = 84  # 84 x 84 image
    
    def __init__(self, memory_length, action_count):
        self.memory_length = memory_length
        self.action_count = action_count
        super(Q, self).__init__(
            l1=F.Convolution2D(memory_length, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            l4=F.Linear(3136, 512, wscale=np.sqrt(2)),
            out=F.Linear(512, self.action_count, initialW=np.zeros((n_act, 512), dtype=np.float32))
        )
    
    def forward(self, state):
        h1 = F.relu(self.l1(state))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        q_value = self.out(h4)
        return q_value


class DQNAgent():
    
    def __init__(self, actions, epsilon=1, memory_length=4):
        self.actions = actions
        self.epsilon = epsilon
        self.q = Q(memory_length, len(actions))
        self._qv = None
        self._state = []
    
    def update_observation(self, observation):
        # some preprocessing is needed
        self._state.append(observation)
        if len(self._state) > self.q.memory_length:
            self._state.pop()
        return observation
    
    def start(self, observation):
        self._state = []
        action = self._act(observation, -1)
        return action
    
    def act(self, observation, reward, done=False, episode_count=-1):
        o = self.update_observation(observation)
        s = self._get_state()
        vs = chainer.Variable(s)
        qv = self.q_func.forward(vs)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, len(self.actions))
        else:
            action = np.argmax(qv)
        
        self._qv = qv

        return action
    
    def _get_state(self):
        state = []

        for  i in range(self.q.memory_length):
            if i < len(self._state):
                state.append(self._state[i])
            else:
                state.append(np.zeros((self.q.SIZE, self.q.SIZE), dtype=np.uint8))
        
        np_state = np.array([state])  # batch_size(1) x memory length x (width x height)
        return np_state
