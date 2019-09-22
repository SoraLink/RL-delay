import tensorflow as tf
import numpy as np

class RNN():
    def __init__(self, sess, scope, hidden_units, max_step_length, input,
                 make_value = 0.37):
        self.input = input
        self.sess = sess
        self.scope = scope
        self.hidden_units = hidden_units
        self.max_step_length = max_step_length
        self.mask_value = make_value
        self.output = self.create_network()

    def create_network(self):
        with tf.variable_scope(self.scope):
            out = tf.keras.layers.Masking(mask_value=self.mask_value)(self.input)
            for i, hidden in enumerate(self.hidden_units):
                out = tf.keras.layers.GRU(hidden,return_sequences=True,return_state=True,
                                              kernel_initializer='ones',recurrent_initializer='zeros',
                                              use_bias=True,bias_initializer='ones')(out)
        return out

    def pad_input(self,data):
        data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=self.max_step_length)
        return data

