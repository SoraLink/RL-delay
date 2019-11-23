import gym
import pybullet
import pybullet_envs

# pybullet.connect(pybullet.DIRECT)
env = gym.make('AntBulletEnv-v0')
env.render(mode="human")
env.reset()
env.render(mode="human")