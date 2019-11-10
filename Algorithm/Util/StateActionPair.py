class StateActionPair:
    def __init__(self, state, actions, reward, done):
        self.state = state
        self.actions = actions
        self.reward = reward
        self.done = done
        self.label = None
        self.predicted_action = None
        self.predicted_state = None

    def set_label(self, label):
        self.label = label

    def set_predicted_action(self, action):
        self.predicted_action = action

    def set_predicted_state(self, state):
        self.predicted_state = state

    def set_reward(self,reward):
        self.reward = reward

    def set_done(self, done):
        self.done = done

    def set_info(self,reward,label,done):
        self.set_done(done)
        self.set_reward(reward)
        self.set_label(label)