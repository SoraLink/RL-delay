class StateActionPair:
    def __init__(self, state=None, actions=None, reward=None):
        self.state = state
        self.actions = actions
        self.reward = reward
        self.done = None
        self.label = None
        self.predicted_action = None
        self.predicted_state = None
        self.value = None
        self.neglogaction = None

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

    def set_info(self, reward, label, done):
        self.set_done(done)
        self.set_reward(reward)
        self.set_label(label)

    def __str__(self):
        string = "state: " + str(self.state) + "\n" +\
        "actions: " + str(self.actions) + "\n" +\
        "reward: " + str(self.reward) + "\n" +\
        "done: " + str(self.done) + "\n" +\
        "label: " + str(self.label) + "\n" +\
        "predicted_action: " + str(self.predicted_action) + "\n" +\
        "predicted_state: " + str(self.predicted_state) + "\n"

        return string