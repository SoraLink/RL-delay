class Pair:
    def __init__(self, state, action, reward, done, next_state):
        self.state = state
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.action = action
