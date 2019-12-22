from Environment.predictedEnv import PredictedEnv
from Environment.registration import EnvRegistry
import time
import tensorflow as tf
from sac.misc.tf_utils import get_default_session
# from tensorflow.python.keras.layers.recurrent import GRUCell

def test():
    sess = get_default_session()
    env = PredictedEnv("HopperPyBulletEnv-v0",2,2,sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()
    while True:
        action = env.env.action_space.sample()
        pair = env.step(action,1)
        print(len(env.data_set.pairs))
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