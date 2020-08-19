import tensorflow as tf
from Model.NeuralNetwork.RNNCell.DRNNCell2 import DRNNCell
from tensorflow.python.keras.layers import RNN
from Algorithm.Util.StateActionPair import StateActionPair
import numpy as np

class Model():
    def __init__(self,
                 sess,
                 rnn_unit,
                 nn_unit,
                 observation_space,
                 action_space,
                 scope,
                 ):
        self.sess = sess
        self.rnn_unit = rnn_unit
        self.nn_unit = nn_unit
        self.scope = scope
        self.observation_space = observation_space
        self.action_space = action_space
        self.output, self.actions, self.init_observation, self.init_state = self.create_network()
        self.target_output = tf.placeholder(tf.float32, [None, self.observation_space])
        # self.init_output = tf.placeholder(tf.float32, [None, self.observation_space])
        self.loss = tf.reduce_mean(tf.square(self.output-self.target_output))
        self.delay_loss = tf.reduce_mean(tf.square(self.init_observation-self.target_output))
        self.update_method = tf.train.AdamOptimizer(3e-6)
        self.update = self.update_method.minimize(self.loss)

    def create_network(self):
        with tf.variable_scope(self.scope):
            actions = tf.placeholder(tf.float32, [None, 1, self.action_space])
            init_observation = tf.placeholder(tf.float32, [None, self.observation_space])
            init_state = tf.placeholder(tf.float32, [None, self.rnn_unit])
            cell = DRNNCell(self.observation_space, self.nn_unit)
            rnn = RNN(cell)
            # output = tf.keras.layers.Masking(mask_value=self.mask_value)(actions)
            output = rnn(inputs=actions, initial_state=[init_observation])
            return output, actions, init_observation, init_state

    def get_param(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_param(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def pad_input(self, data):
        data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=self.delay, value=self.mask_value)
        return data

    def run(self, pair):
        predicted_state = pair.state
        for action in pair.actions:
            predicted_state = self.sess.run(self.output,feed_dict={
                self.actions : [[action]],
                self.init_observation : [predicted_state]
            })[0]
        pair.set_predicted_state(predicted_state)
        return pair

    def train(self, pairs):
        actions, states, next_states = extract(pairs)
        # _ , loss = self.sess.run((self.update, self.loss), feed_dict={
        #     self.actions: actions,
        #     self.output: predicted_states,
        #     self.target_output: target_states,
        #     self.init_observation : states,
        #     self.init_state : [np.zeros(self.rnn_unit)]
        # })
        # print(self.actions, np.array(actions).shape)
        _ , loss = self.sess.run((self.update, self.loss), feed_dict={
            self.actions: actions,
            self.target_output: next_states,
            self.init_observation : states,
            # self.init_state : [np.zeros(self.rnn_unit)]
        })
        # print(loss)
        return loss

def extract(pairs):
    actions = list()
    actions += [[pair.action] for pair in pairs]
    states = list()
    states += [pair.state for pair in pairs]
    next_states = list()
    next_states += [pair.next_state for pair in pairs]
    return [actions, states, next_states]