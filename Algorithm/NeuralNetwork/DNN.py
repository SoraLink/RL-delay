import numpy as np
import tensorflow as tf
class DNN():
    def __init__(self,sess,scope,hidden_units,input):
        self.input = input
        self.sess = sess
        self.scope = scope
        self.hidden_units = hidden_units
        self.output = self.create_network()

    def create_network(self):
        with tf.variable_scope(self.scope):
            out = self.input
            for i, hidden in enumerate(self.hidden_units):
                out = tf.nn.tanh(tf.layers.dense(out, hidden, name="fc%i" % (i + 1),
                                                 kernel_initializer=tf.random_normal_initializer(stddev=1.0,mean=0)))
        return out

    def get_param(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_param(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)