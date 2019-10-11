import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell
from tensorflow.python.keras import activations
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.keras.layers import Dense

from tensorflow.nn.rnn_cell import LSTMCell

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class DRNNCell(GRUCell):

    def __init__(self,num_units, transition_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(DRNNCell, self).__init__(_reuse = reuse, name = name,dtype=dtype,**kwargs)
        self._num_units = num_units
        self._transition_units = transition_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                             str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(self._bias_initializer
                         if self._bias_initializer is not None else
                         init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(self._bias_initializer
                         if self._bias_initializer is not None else
                         init_ops.zeros_initializer(dtype=self.dtype)))
        self.built = True

    def call(self, action, state):
        _check_rnn_cell_input_dtypes([action,state])
        inputs = array_ops.slice(state,[0,0],[-1,self._transition_units])
        state = array_ops.slice(state,[0,self._transition_units],[-1,self._num_units])
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        outputs = Dense(self._transition_units)(new_h)
        outputs = Dense(self._transition_units)(outputs)
        next_state = array_ops.concat([outputs,new_h],1)
        return outputs, next_state


def _check_rnn_cell_input_dtypes(inputs):
  """Check whether the input tensors are with supported dtypes.

  Default RNN cells only support floats and complex as its dtypes since the
  activation function (tanh and sigmoid) only allow those types. This function
  will throw a proper error message if the inputs is not in a supported type.

  Args:
    inputs: tensor or nested structure of tensors that are feed to RNN cell as
      input or state.

  Raises:
    ValueError: if any of the input tensor are not having dtypes of float or
      complex.
  """
  for t in nest.flatten(inputs):
    _check_supported_dtypes(t.dtype)


def _check_supported_dtypes(dtype):
  if dtype is None:
    return
  dtype = dtypes.as_dtype(dtype)
  if not (dtype.is_floating or dtype.is_complex):
    raise ValueError("RNN cell only supports floating point inputs, "
                     "but saw dtype: %s" % dtype)





