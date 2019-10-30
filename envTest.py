from Environment.registration import EnvRegistry
import time

def test():
    env = EnvRegistry("CartPole-v1",20,20)
    env.start()
    while True:
        action = env.action_space.sample()
        feedback = env.step(action)
        if feedback is None:
            continue
        else:
            observation, reward, done, info, action = feedback
            print("Observation: ", observation)
            print("reward: ", reward)
            print("done: ", done)
            print("info: ", info)
            print("action: ", action)
            time.sleep(0.01)
            if done:
                env.restart()



if __name__ == '__main__':
    test()