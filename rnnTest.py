from Environment.registration import EnvRegistry
import time
import tensorflow as tf
from Model.NeuralNetwork.RNNCell.DRNNCell import DRNNCell
from tensorflow.keras.layers import RNN
from tensorflow.keras.layers import GRU

def test():
    a = tf.placeholder(tf.float32, (None,10,10))
    initial_state1 = tf.placeholder(tf.float32, (None,7))
    initial_state2 = tf.placeholder(tf.float32, (None,7))
    cell = DRNNCell(7,transition_units=10)
    output = RNN(cell)
    # output = GRU(5)
    output = output(inputs = a, initial_state = [initial_state1, initial_state2])
    # output = output(inputs=a, initial_state=initial_state1)


if __name__ == '__main__':
    test()