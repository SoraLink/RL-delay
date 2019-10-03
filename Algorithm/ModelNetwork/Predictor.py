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
            GRU_output = RNN(self.sess, self.scope + "_GRU", self.GRU_hidden_units, self.max_step_length, self.state)\
                .output
            fc_input = tf.concat([self.action, GRU_output], 1)
            output = DNN(self.sess, self.scope + '_DNN', self.NN_hidden_units, fc_input).output
        return output

    def get_param(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

    def get_trainable_param(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    # from tensorflow.python.ops.rnn import _transpose_batch_time
    # import tensorflow as tf
    #
    # def sampling_rnn(self, cell, initial_state, input_, seq_lengths):
    #
    #     # raw_rnn expects time major inputs as TensorArrays
    #     max_time = ...  # this is the max time step per batch
    #     inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time, clear_after_read=False)
    #     inputs_ta = inputs_ta.unstack(_transpose_batch_time(input_))  # model_input is the input placeholder
    #     input_dim = input_.get_shape()[-1].value  # the dimensionality of the input to each time step
    #     output_dim = ...  # the dimensionality of the model's output at each time step
    #
    #     def loop_fn(time, cell_output, cell_state, loop_state):
    #         """
    #         Loop function that allows to control input to the rnn cell and manipulate cell outputs.
    #         :param time: current time step
    #         :param cell_output: output from previous time step or None if time == 0
    #         :param cell_state: cell state from previous time step
    #         :param loop_state: custom loop state to share information between different iterations of this loop fn
    #         :return: tuple consisting of
    #           elements_finished: tensor of size [bach_size] which is True for sequences that have reached their end,
    #             needed because of variable sequence size
    #           next_input: input to next time step
    #           next_cell_state: cell state forwarded to next time step
    #           emit_output: The first return argument of raw_rnn. This is not necessarily the output of the RNN cell,
    #             but could e.g. be the output of a dense layer attached to the rnn layer.
    #           next_loop_state: loop state forwarded to the next time step
    #         """
    #         if cell_output is None:
    #             # time == 0, used for initialization before first call to cell
    #             next_cell_state = initial_state
    #             # the emit_output in this case tells TF how future emits look
    #             emit_output = tf.zeros([output_dim])
    #         else:
    #             # t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
    #             # here you can do whatever ou want with cell_output before assigning it to emit_output.
    #             # In this case, we don't do anything
    #             next_cell_state = cell_state
    #             emit_output = cell_output
    #
    #             # check which elements are finished
    #         elements_finished = (time >= seq_lengths)
    #         finished = tf.reduce_all(elements_finished)
    #
    #         # assemble cell input for upcoming time step
    #         current_output = emit_output if cell_output is not None else None
    #         input_original = inputs_ta.read(time)  # tensor of shape (None, input_dim)
    #
    #         if current_output is None:
    #             # this is the initial step, i.e. there is no output from a previous time step, what we feed here
    #             # can highly depend on the data. In this case we just assign the actual input in the first time step.
    #             next_in = input_original
    #         else:
    #             # time > 0, so just use previous output as next input
    #             # here you could do fancier things, whatever you want to do before passing the data into the rnn cell
    #             # if here you were to pass input_original than you would get the normal behaviour of dynamic_rnn
    #             next_in = current_output
    #
    #         next_input = tf.cond(finished,
    #                              lambda: tf.zeros([self.batch_size, input_dim], dtype=tf.float32),  # copy through zeros
    #                              lambda: next_in)  # if not finished, feed the previous output as next input
    #
    #         # set shape manually, otherwise it is not defined for the last dimensions
    #         next_input.set_shape([None, input_dim])
    #
    #         # loop state not used in this example
    #         next_loop_state = None
    #         return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)
    #
    # outputs_ta, last_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    # outputs = _transpose_batch_time(outputs_ta.stack())
    # final_state = last_state
    #
    # return outputs, final_state