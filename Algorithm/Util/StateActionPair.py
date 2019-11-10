class StateActionPair:
    def __init__(self, state, actions, reward, done):
        self.state = state
        self.actions = actions
        self.reward = reward
        self.done = done
        self.label = None
        self.predicted_action = None

    def set_label(self, label):
        self.label = label

    def set_predicted_action(self, action):
        self.predicted_action = action