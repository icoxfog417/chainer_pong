import os
import numpy as np
from chainer import Chain
from chainer import Variable
from chainer import cuda
from chainer import serializers
import chainer.functions as F
from model.agent import Agent


class Q(Chain):
    """
    You want to optimize this function to determine the action from state (state is represented by CNN vector)
    """

    SIZE = 80  # 80 x 80 image
    
    def __init__(self, n_history, n_action, on_gpu=False):
        self.n_history = n_history
        self.n_action = n_action
        self.on_gpu = on_gpu
        super(Q, self).__init__(
            l1=F.Convolution2D(n_history, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(32, 64, ksize=3, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            l4=F.Linear(3136, 512, wscale=np.sqrt(2)),
            out=F.Linear(512, self.n_action, initialW=np.zeros((n_action, 512), dtype=np.float32))
        )
    
    def __call__(self, state: np.ndarray):
        _state = self.to_gpu(state)
        s = Variable(_state)
        h1 = F.relu(self.l1(state))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        q_value = self.out(h4)
        return q_value
    
    def to_gpu(self, arr):
        return arr if not self.on_gpu else cuda.to_gpu(arr)


class DQNAgent(Agent):
    
    def __init__(self, actions, epsilon=1, n_history=4, on_gpu=False, model_path="", load_if_exist=True):
        self.actions = actions
        self.epsilon = epsilon
        self.q = Q(n_history, len(actions), on_gpu)
        self._state = []
        self._observations = [
            np.zeros((self.q.SIZE, self.q.SIZE), np.float32), 
            np.zeros((self.q.SIZE, self.q.SIZE), np.float32)
        ]  # now & pre
        self.last_action = 0
        self.model_path = model_path if model_path else os.path.join(os.path.dirname(__file__), "./store")
        if not os.path.exists(self.model_path):
            print("make directory to store model at {0}".format(self.model_path))
            os.mkdir(self.model_path)
        else:
            models = self.get_model_files()
            if load_if_exist and len(models) > 0:
                serializers.load_npz(os.path.join(self.model_path, models[-1]), self.q)  # use latest model
    
    def _update_state(self, observation):
        formatted = self._format(observation)
        state = np.maximum(formatted, self._observations[0])
        self._state.append(state)
        if len(self._state) > self.q.n_history:
            self._state.pop(0)
        return formatted
    
    @classmethod
    def _format(cls, image):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        im = image[35:195] # crop
        im = im[::2,::2,0] # downsample by factor of 2
        im[im == 144] = 0 # erase background (background type 1)
        im[im == 109] = 0 # erase background (background type 2)
        im[im != 0] = 1 # everything else (paddles, ball) just set to 1
        return im.astype(np.float32)

    def start(self, observation):
        self._state = []
        self._observations = [
            np.zeros((self.q.SIZE, self.q.SIZE), np.float32), 
            np.zeros((self.q.SIZE, self.q.SIZE), np.float32)
        ]
        self.last_action = 0

        action = self.act(observation, 0)
        return action
    
    def act(self, observation, reward):
        o = self._update_state(observation)
        s = self.get_state()
        qv = self.q(np.array([s])) # batch size = 1

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, len(self.actions))
        else:
            action = np.argmax(qv.data[-1])
        
        self._observations[-1] = self._observations[0].copy()
        self._observations[0] = o
        self.last_action = action

        return action

    def get_state(self):
        state = []
        for  i in range(self.q.n_history):
            if i < len(self._state):
                state.append(self._state[i])
            else:
                state.append(np.zeros((self.q.SIZE, self.q.SIZE), dtype=np.float32))
        
        np_state = np.array(state)  # n_history x (width x height)
        return np_state
    
    def save(self, index=0):
        fname = "pong.model" if index == 0 else "pong_{0}.model".format(index)
        path = os.path.join(self.model_path, fname)
        serializers.save_npz(path, self.q)
    
    def get_model_files(self):
        files = os.listdir(self.model_path)
        model_files = []
        for f in files:
            if f.startswith("pong") and f.endswith(".model"):
                model_files.append(f)
        
        model_files.sort()
        return model_files

    @classmethod
    def model_path(cls, fn="pong.model"):
        _fn = fn if fn else "pong.model"
        return os.path.join(os.path.dirname(__file__), "models/" + _fn)
