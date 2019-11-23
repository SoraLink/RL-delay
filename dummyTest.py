import gym
import pybullet
import pybullet_envs
import time

# pybullet.connect(pybullet.DIRECT)
env = gym.make('AntBulletEnv-v0')
env.render(mode="human")
env.reset()
env.render(mode="human")
while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    # print(obs, rew, done, info)
    env.render(mode = 'human')
    time.sleep(0.01)