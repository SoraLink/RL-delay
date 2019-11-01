from Environment.registration import EnvRegistry
class StateActionPair():
    def __init__(self, state, actions):
        self.state = state
        self.actions = actions

    def get_actions(self):
        return self.actions

    def get_state(self):
        return self.state

    def add_action(self,action):
        self.actions.append(action)