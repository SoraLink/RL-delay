class StateActionPair:
    def __init__(self, state=None, actions=None, reward=None,
                done = None, label = None, predicted_action = None,
                predicted_state = None, value = None, neglogaction = None
                ):
        self.state = state
        self.actions = actions
        self.reward = reward
        self.done = done
        self.label = label
        self.predicted_action = predicted_action
        self.predicted_state = predicted_state
        self.value = value
        self.neglogaction = neglogaction
        self.next_state = None
        self.action_m = None

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

    def set_info(self, reward, label):
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