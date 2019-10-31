from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import GRUCell
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTMCell
from tensorflow.python.keras.layers import GRU
import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class DRNNCell(GRUCell):
    def __init__(self,
               units,
               transition_units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               reset_after=False,
               **kwargs):
        super(DRNNCell, self).__init__(
            units,
            activation,
            recurrent_activation,
            use_bias,
            kernel_initializer,
            recurrent_initializer,
            bias_initializer,
            kernel_regularizer,
            recurrent_regularizer,
            bias_regularizer,
            kernel_constraint,
            recurrent_constraint,
            bias_constraint,
            dropout,
            recurrent_dropout,
            implementation,
            reset_after,
            **kwargs
        )
        self._transition_units = transition_units

    def call(self, action, states, training=None):
        h_tm1 = states[0]  # previous memory
        inputs = states[1] # previous output
        # h_tm1, inputs = tf.split(states[0], self.units)
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = array_ops.unstack(self.bias)

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = K.dot(inputs_z, self.kernel[:, :self.units])
            x_r = K.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
            x_h = K.dot(inputs_h, self.kernel[:, self.units * 2:])

            if self.use_bias:
                x_z = K.bias_add(x_z, input_bias[:self.units])
                x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
                x_h = K.bias_add(x_h, input_bias[self.units * 2:])

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
            recurrent_r = K.dot(h_tm1_r,
                                self.recurrent_kernel[:, self.units:self.units * 2])
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
                recurrent_r = K.bias_add(recurrent_r,
                                         recurrent_bias[self.units:self.units * 2])

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1_h,
                                    self.recurrent_kernel[:, self.units * 2:])

            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = K.bias_add(matrix_x, input_bias)

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            x_h = matrix_x[:, 2 * self.units:]

            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units:2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:]
            else:
                recurrent_h = K.dot(r * h_tm1,
                                    self.recurrent_kernel[:, 2 * self.units:])

            hh = self.activation(x_h + recurrent_h)
        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        outputs = array_ops.concat([h, action],0)
        outputs = Dense(self._transition_units)(outputs)
        outputs = Dense(self._transition_units)(outputs)
        # h = tf.concat([h, outputs], 0)
        return outputs, [h, outputs]
        # _check_rnn_cell_input_dtypes([action,state])
        # inputs = array_ops.slice(state,[0,0],[-1,self._transition_units])
        # state = array_ops.slice(state,[0,self._transition_units],[-1,self._num_units])
        # gate_inputs = math_ops.matmul(
        #     array_ops.concat([inputs, state], 1), self._gate_kernel)
        # gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        #
        # value = math_ops.sigmoid(gate_inputs)
        # r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        #
        # r_state = r * state
        #
        # candidate = math_ops.matmul(
        #     array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        # candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        #
        # c = self._activation(candidate)
        # new_h = u * state + (1 - u) * c
        # outputs = Dense(self._transition_units)(new_h)
        # outputs = Dense(self._transition_units)(outputs)
        # next_state = array_ops.concat([outputs,new_h],1)
        # return outputs, next_state


