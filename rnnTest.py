# from Environment.registration import EnvRegistry
# import time
import tensorflow as tf
from Model.NeuralNetwork.RNNCell.DRNNCell import DRNNCell
from tensorflow.python.keras.layers import RNN
from tensorflow.python.keras.layers import GRU

def test():
    a = tf.placeholder(tf.float32, (None,10,2))
    initial_state1 = tf.placeholder(tf.float32, (None,32))
    initial_state2 = tf.placeholder(tf.float32, (None,4))
    cell = DRNNCell(32,30,4)
    output = RNN(cell)
    # output = GRU(5)
    output = output(inputs = a, initial_state = [initial_state1, initial_state2])
    # output = output(inputs=a, initial_state=initial_state1)
    print(output)


if __name__ == '__main__':
    test()