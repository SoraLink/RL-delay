from Model.NeuralNetwork.RNN import RNN
from Model.NeuralNetwork.DNN import DNN
import tensorflow as tf

class Predictor():
    def __init__(self, sess, scope, GRU_hidden_units, max_step_length, NN_hidden_units, action_dim, state_dim):
        self.sess = sess
        self.scope = scope
        self.GRU_hidden_units = GRU_hidden_units
        self.max_step_length = max_step_length
        self.state = tf.placeholder(shape=(None,state_dim),dtype=tf.float32)
        self.action = tf.placeholder(shape=(None,action_dim),dtype=tf.float32)
        self.NN_hidden_units = NN_hidden_units
        self.output = self.create_model()

    def create_model(self):
        with tf.variable_scope(self.scope):
            GRU_output = RNN(self.sess, self.scope + "_GRU", self.GRU_hidden_units, self.max_step_length, self.state).output
            fc_input = tf.concat([self.action,GRU_output],1)
            output = DNN(self.sess, self.scope + '_DNN', self.NN_hidden_units, fc_input).output
        return output

    def get_param(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_param(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)