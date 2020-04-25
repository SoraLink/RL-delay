import tensorflow as tf
from Model.NeuralNetwork.RNNCell.DRNNCell2 import DRNNCell
from tensorflow.python.keras.layers import RNN
from Algorithm.Util.Pair import Pair
import numpy as np

class Model():
    ### Only us fcnn to predict next state ###
    def __init__(self,
                 sess,
                 nn_unit,
                 observation_space,
                 action_space,
                 scope,
                 ):
        self.sess = sess
        self.nn_unit = nn_unit
        self.scope = scope
        self.observation_space = observation_space
        self.action_space = action_space
        self.output, self.action, self.state= self.create_network()
        self.next_state = tf.placeholder(tf.float32, [None, self.observation_space])
        self.loss = tf.reduce_mean(tf.square(self.output-self.next_state))
        self.update_method = tf.train.AdamOptimizer(3e-6)
        self.update = self.update_method.minimize(self.loss)

    def create_network(self):
        with tf.variable_scope(self.scope):
            action = tf.placeholder(tf.float32, [None, self.action_space])
            state = tf.placeholder(tf.float32, [None, self.observation_space])
            out = tf.concat([action, state],axis=1)
            for i, hidden in enumerate(self.nn_unit):
                out = tf.nn.tanh(tf.layers.dense(out, hidden, name="fc%i" % (i+1),
                                                    kernel_initializer=tf.random_normal_initializer(stddev=1.0,mean=0)))
            out = tf.nn.relu(tf.layers.dense(out, self.observation_space, name="fc%i" % (len(self.nn_unit)+1),
                                                    kernel_initializer=tf.random_normal_initializer(stddev=1.0,mean=0)))
            return out, action, state

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
                self.action : [action],
                self.state : [predicted_state]
            })
            pair.set_predicted_state(predicted_state[0])
            return pair

    def train(self, pairs):
        actions, states, next_states = extract(pairs)
        _ , loss, predicted_state = self.sess.run((self.update, self.loss, self.output), feed_dict={
            self.action: actions,
            self.next_state: next_states,
            self.state : states,
        })
        print(loss)
        return loss

def extract(pairs):
    actions = list()
    actions += [pair.action for pair in pairs]
    states = list()
    states += [pair.state for pair in pairs]
    next_states = list()
    next_states += [pair.next_state for pair in pairs]
    return [actions, states, next_states]