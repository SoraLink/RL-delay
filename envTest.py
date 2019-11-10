from Environment.registration import EnvRegistry
import time

def test():
    env = EnvRegistry("CartPole-v1",2,2)
    env.start()
    # print(type(env.env.spec))
    # while True:
    #     action = env.action_space.sample()
    #     feedback = env.step(action)
    #     if feedback is None:
    #         continue
    #     else:
    #         pair = feedback
    #         print("Observation: ", pair.state)
    #         print("reward: ", pair.reward)
    #         print("done: ", pair.done)
    #         print("label: ", pair.label)
    #         print("actions: ", pair.actions)
    #         time.sleep(0.01)
    #         if pair.done :
    #             env.restart()


if __name__ == '__main__':
    test()

    # self.state = state
    # self.actions = actions
    # self.reward = reward
    # self.done = done
    # self.label = None
    # self.predicted_action = None
    # self.predicted_state = None