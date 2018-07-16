from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell_impl

from pprint import pprint

def static_rnn(cell, inputs, init_state=None, dtype=tf.float32, keep_all_states=False,
               scope=None):
    """ The basic rnn encoder
        Args:
            cell: a RNN cell
            inputs: inputs to the rnn, of shape [(batch_size, hidden_size)], length equals to rnn length
            init_state: the inital state of the RNN, if not specified, the init_state would be all zero, of shape cell.state.get_shape()
        Returns:
            outs: the list of output emits from the network, of shape [(batch_size, hidden_size)], length equals to rnn length
            state: the last state of the rnn
    """
    with tf.variable_scope(scope or "static_rnn") as scope:

        batch_size = tf.shape(inputs[0])[0]

        if init_state is None:
            init_state = cell.zero_state(batch_size, dtype)

        step_num = len(inputs)

        outs = []
        if keep_all_states:
            all_states = []

        state = init_state
        for i in range(step_num):
            (output, state) = cell(inputs[i], state)
            outs.append(output)
            if keep_all_states:
                all_states.append(state)

    if keep_all_states:
        return outs, all_states
    else:
        return outs, state


def dynamic_rnn(cell, inputs, masks, 
                init_state=None, keep_all_states=False,
                dtype=tf.float32, scope=None):
    """ Dynamic rnn that supports state forwarding when paddings are included in the input """
    with tf.variable_scope(scope or "dynamic_rnn") as scope:

        batch_size = tf.shape(inputs[0])[0]

        if init_state is None:
            init_state = cell.zero_state(batch_size, dtype)

        step_num = len(inputs)

        outs = []

        if keep_all_states:
            all_states = []

        state = init_state
        for i in range(step_num):
            
            (output, new_state) = cell(inputs[i], state)

            # copy through the old state the current input is a padding token
            if isinstance(cell, tf.contrib.rnn.LSTMCell):
                state = f_apply_lstm_state(new_state, state, lambda s1, s2: tf.where(masks[i], s1, s2))
            elif isinstance(cell, tf.contrib.rnn.MultiRNNCell):
                state = f_apply_multirnn_lstm_state(new_state, state, lambda s1, s2: tf.where(masks[i], s1, s2))

            # emit zero output is the current token is 
            output = tf.where(masks[i], output, tf.zeros(tf.shape(output), dtype))
            
            outs.append(output)

            if keep_all_states:
                all_states.append(state)

    if keep_all_states:
        return outs, all_states
    else:
        return outs, state


def bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, masks,
                              init_fw_state=None, keep_all_states=False,
                              dtype=tf.float32, scope=None):
    """ The bidirectional rnn
        Args:
            fw_cell: a RNN cell
            bw_cell: another RNN cell for the backward pass
            inputs: inputs to the rnn, of shape [(batch_size, hidden_size)], length equals to rnn length
            init_state: the inital state of the RNN, if not specified, the init_state would be all zero, of shape cell.state.get_shape()
        Returns:
            outs: the list of output emits from the network, of shape [(batch_size, hidden_size * 2)], length equals to rnn length
            fw_state: the last state of the fw rnn
            bw_state: the last state of the backward rnn
    """
    with tf.variable_scope(scope or "bidirectional_rnn_encoder"):

        with tf.variable_scope("fw") as fw_scope:
            fw_outs, fw_state = dynamic_rnn(fw_cell, inputs, masks, 
                                            init_state=init_fw_state, keep_all_states=keep_all_states, 
                                            dtype=dtype, scope=fw_scope)

        with tf.variable_scope("bw") as bw_scope:
            # reverse the input to the backward encoder
            reversed_inputs = inputs[::-1]
            reversed_masks = masks[::-1]
            init_bw_state = fw_state if not keep_all_states else fw_state[-1]
            bw_outs, bw_state = dynamic_rnn(bw_cell, reversed_inputs, reversed_masks, 
                                            init_state=init_bw_state, keep_all_states=keep_all_states,
                                            dtype=dtype, scope=bw_scope)

    reversed_bw_outs = bw_outs[::-1]
    outs = tf.concat([fw_outs, reversed_bw_outs], -1)

    return outs, fw_state, bw_state


def bidirectional_static_rnn(fw_cell, bw_cell, inputs, masks,
                            init_fw_state=None, keep_all_states=False,
                            dtype=tf.float32, scope=None):
    """ The bidirectional rnn
        Args:
            fw_cell: a RNN cell
            bw_cell: another RNN cell for the backward pass
            inputs: inputs to the rnn, of shape [(batch_size, hidden_size)], length equals to rnn length
            init_state: the inital state of the RNN, if not specified, the init_state would be all zero, of shape cell.state.get_shape()
        Returns:
            outs: the list of output emits from the network, of shape [(batch_size, hidden_size * 2)], length equals to rnn length
            fw_state: the last state of the fw rnn
            bw_state: the last state of the backward rnn
    """
    with tf.variable_scope(scope or "bidirectional_rnn_encoder"):

        with tf.variable_scope("fw") as fw_scope:
            fw_outs, fw_state = static_rnn(fw_cell, inputs,
                                            init_state=init_fw_state,
                                            dtype=dtype, scope=fw_scope,
                                            keep_all_states=keep_all_states)

        with tf.variable_scope("bw") as bw_scope:
            # reverse the input to the backward encoder
            reversed_inputs = inputs[::-1]
            reversed_masks = masks[::-1]
            init_bw_state = fw_state if not keep_all_states else fw_state[-1]
            bw_outs, bw_state = static_rnn(bw_cell, reversed_inputs,
                                            init_state=init_bw_state,
                                            dtype=dtype, scope=bw_scope,
                                            keep_all_states=keep_all_states)

    reversed_bw_outs = bw_outs[::-1]
    outs = tf.concat([fw_outs, reversed_bw_outs], -1)

    return outs, fw_state, bw_state


def f_apply_multirnn_lstm_state(state1, state2, f):
    """ Given two multirnn lstm states, merge them into a new state of the same shape, merged by concatation and then projection
        Args:
            state1: the first state to mergem, of shape (s1, s2, s3, ...), 
                    each s is of shape LSTMStateTuple(c,h), h,c are of shape (batch_size, hidden_size)
            state2: the second state to merge, shape same as the first
            w: the projection weight, of shape (hidden_size * 2, hidden_size)
            b: the projection bias, of shape (hidden_size,)
        Returns:
            the merged states
    """
    new_state = []
    for i in range(len(state1)):
        new_state.append(f_apply_lstm_state(state1[i], state2[i], f))
    return tuple(new_state)


def f_apply_lstm_state(state1, state2, f):
    """ merge two lstm states into one of the same shape as either or them, merged by concatation and then projection
        Args:
            state1, state2: two states to be merged, of shape LSTMStateTuple(c,h), h,c are of shape (batch_size, hidden_size)
            w: the projection weight, of shape (hidden_size * 2, hidden_size)
            b: the projection bias, of shape (hidden_size,)
        Returns:
            the merged state, of shape (batch_size, hidden_size)
    """
    return rnn_cell_impl.LSTMStateTuple(f(state1.c, state2.c), f(state1.h, state2.h))


def merge_multirnn_lstm_state(states, w, b):
    """ Given two multirnn lstm states, merge them into a new state of the same shape, merged by concatation and then projection
        Args:
            state1: the first state to mergem, of shape (s1, s2, s3, ...), 
                    each s is of shape LSTMStateTuple(c,h), h,c are of shape (batch_size, hidden_size)
            state2: the second state to merge, shape same as the first
            w: the projection weight, of shape (hidden_size * 2, hidden_size)
            b: the projection bias, of shape (hidden_size,)
        Returns:
            the merged states
    """
    new_state = []
    for i in range(len(states[0])):
        new_state.append(merge_lstm_states([s[i] for s in states], w, b))
    return tuple(new_state)


def merge_lstm_states(states, w, b):
    """ merge two lstm states into one of the same shape as either or them, merged by concatation and then projection
        Args:
            state1, state2: two states to be merged, of shape LSTMStateTuple(c,h), h,c are of shape (batch_size, hidden_size)
            w: the projection weight, of shape (hidden_size * k, hidden_size)
            b: the projection bias, of shape (hidden_size,)
        Returns:
            the merged state, of shape (batch_size, hidden_size)
    """
    new_c = tf.add(tf.matmul(tf.concat([x.c for x in states], -1), w), b)
    new_h = tf.add(tf.matmul(tf.concat([x.h for x in states], -1), w), b)
    return rnn_cell_impl.LSTMStateTuple(new_c, new_h)

