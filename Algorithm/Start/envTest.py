from Environment.registration import EnvRegistry

def test():
    env = EnvRegistry("CartPole-v1",20,20)
    env.start()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info, action = env.step(action)
        print("Observation: ", observation)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        print("action: ", action)



if __name__ == '__main__':
    test()