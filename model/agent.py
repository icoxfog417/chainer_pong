class Agent():

    def __init__(self, actions):
        self.actions = actions

    def start(self, observation):
        return 0  # default action

    def act(self, observation, reward):
        return 0  # default action
    
    def end(self, observation, reward):
        pass

    def report(self, episode):
        return ""
