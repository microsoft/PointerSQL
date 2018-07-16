from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops

from model.util import Vocabulary
from model.rnn import merge_multirnn_lstm_state
from model.rnn import bidirectional_static_rnn

from pprint import pprint

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest


linear = rnn_cell_impl._linear
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def __linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None, weights=None, biases=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = variable_scope.get_variable_scope()
  with variable_scope.variable_scope(scope) as outer_scope:
    if weights is None:
        weights = variable_scope.get_variable(
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with variable_scope.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      if biases is None:
          biases = variable_scope.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)


class DecoderType(object):
    """ This class defines possible types for the decoder,
        Pointer means that the decoder should decode by pointing to a value in the input sequence
        Projector means that the decoder should decode by projecting the value and lookup in the dictionary
    """
    Pointer = "pointer"
    Projector = "projector"

    def __init__(self, ty, mask_name=None, mask_fn=None):
        """ Create a decoder type object, dec_ty defines whether to point or project, 
            type defines which parts of attention states should be ignored (in decoding phase)"""
        self.ty = ty
        self.mask_name = mask_name
        self.mask_fn = mask_fn

    def __str__(self):
        return "{}:{}".format(self.ty, self.mask_name)

    @staticmethod
    def from_regex(s, abbrv_dict=None, maxlen=1000):
        if abbrv_dict is None:
            abbrv_dict = {
                "w": DecoderType(DecoderType.Projector),
                "p": DecoderType(DecoderType.Pointer)
            }

        result = []
        i = 0
        lb = -1

        while len(result) < maxlen:
            if i == len(s):
                break
            if s[i] == "(":
                lb = i
            elif s[i] == ")":
                pass
            elif s[i] == "*":
                i = lb
            else:
                # it is supposed to be a symbol in the decoder
                result.append(abbrv_dict[s[i]])
            i += 1

        return result


def pointer_network_decoder(cell,
                            decoder_type, # specification of the cell
                            decoder_inputs, 
                            init_state,
                            attention_states,
                            num_decoder_symbols,
                            loop_function=None,
                            dtype=tf.float32,
                            scope=None, weights=None):
    """ A decoder with attention mechemnism
        Arguments:
            cell: a RNN cell
            decoder_inputs: inputs to the decoder, of shape [(batch_size, embedding_size)], 
                            the list length is same as the size of the decoder
            init_state: the initial hidden state for the decoder, of shape (hidden_size, ) 
            attention_states: states from the encoder that the attention header should look at, 
                             of shape (batch_size, encoder_length, hidden_size), where encoder_length is the size of encoder
            num_decoder_symbols: the number of symbols in the decoder vocabulary
            loop_function: used to deal with the problem of feed_previous, if the loop_function is None (training time), 
                           the input to the t-th cell will be decoder_inputs[t], otherwise the input will be the output of (t-1)-th cell
        Returns:
            A tuple of form (outs, state): 
                outs is a list, whose length is same as the decoder size
                state is a tensor produced by the last unit of the decoder
    """

    with variable_scope.variable_scope(scope or "pointer_network_decoder") as scope:
        batch_size = tf.shape(attention_states)[0]
        encoder_length = attention_states.get_shape()[1]
        
        input_size = decoder_inputs[0].get_shape()[1]
        
        hidden_size = cell.output_size # size of the hidden state
        attn_size = attention_states.get_shape()[2] # size of the attention vector

        if weights is not None:
            W1 = weights['decoder_W1']
            W2 = weights['decoder_W2']
            v = weights['decoder_v']
            Wout = weights['decoder_Wout'] if 'decoder_Wout' in weights else None
            Bout = weights['decoder_Bout'] if 'decoder_Bout' in weights else None
            Win = weights['decoder_Win'] if 'decoder_Win' in weights else None
            Bin = weights['decoder_Bin'] if 'decoder_Bin' in weights else None
        else:
            # weights for attention: W1 is the weight for input hidden states, W2 is the weight for current output
            # the second dimension is the size of the engery function for attentions
            W1 = tf.get_variable("W1", [attn_size, hidden_size], initializer=tf.random_normal_initializer())
            W2 = tf.get_variable("W2", [hidden_size, hidden_size], initializer=tf.random_normal_initializer())
            v = tf.get_variable("v", [hidden_size])
            Wout, Bout, Win, Bin =None, None, None, None

        # copy W1, W2, v for batch_size times so that we can use mat multiply later
        copied_W1 = tf.tile(tf.expand_dims(W1, 0), [batch_size, 1, 1])
        copied_W2 = tf.tile(tf.expand_dims(W2, 0), [batch_size, 1, 1])
        copied_v = tf.tile(tf.expand_dims(tf.expand_dims(v, -1), 0), [batch_size, 1, 1])

        step_num = len(decoder_inputs)

        outs = []
        prev = None

        # the initial attention value
        attn = tf.zeros(shape=[batch_size, attn_size], dtype=dtype)

        state = init_state
        for i in range(step_num):

            inp = decoder_inputs[i]
            if loop_function is not None and prev is not None:
                # indicate the type of prev
                inp = loop_function(prev, decoder_type[i-1])

            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            with variable_scope.variable_scope("input_linear"):
                inp = __linear([inp] + [attn], input_size, True, weights=Win, biases=Bin)

            (output, state) = cell(inp, state)

            tiled_out = tf.transpose(tf.stack([output for x in range(encoder_length)]), [1, 0, 2])

            ## corresponding formula: u = tanh(h * W1 + d * W2) * v
            ## we use copied W1 and W2 because we want to element-wisely perform matrix multiplication
            u = tf.matmul(tf.tanh(tf.matmul(attention_states, copied_W1) + tf.matmul(tiled_out, copied_W2)), copied_v)

            ## corresponding formula: a = softmax(u) 
            a = tf.nn.softmax(u, 1)

            # perform a point-wise multiply and then perform a summation over the dimension to get the proper distribution
            attn = tf.reduce_sum(tf.multiply(attention_states, a), 1)

            with variable_scope.variable_scope("output_linear"):
                # this makes the dimension of the output variable 
                new_out = __linear([output] + [attn], num_decoder_symbols, True, weights=Wout, biases=Bout)
            
            # output of the current decoder depends on the cell type
            if decoder_type[i].ty is DecoderType.Pointer:
                new_out = tf.squeeze(u, axis=-1)
            elif decoder_type[i].ty is DecoderType.Projector:
                # project to the dimension of decoder symbol for later decoding
                pass
            else:
                raise Exception('not a expected type')

            outs.append(new_out)
            if loop_function is not None:
                prev = new_out

    return outs, state


def pointer_network(enc_embeds,
                    dec_embeds,
                    fw_enc_cell,
                    bw_enc_cell,
                    dec_cell,
                    decoder_type, # specification of the cell
                    num_decoder_symbols,
                    encoder_masks,
                    feed_prev=None,
                    loop_function=None,
                    multi_encoders=None,
                    encoder_merge_method="sequential", # 
                    dtype=None,
                    scope=None, weights=None):
    """ Create a pointer netowrk
        Args:
            enc_embeds: encoder inputs to the pointer network, of shape (batch_size, embedding_size)
            dec_embeds: decoder inputs to the pointer network, of shape (batch_size, embedding_size)
            cell: the template for encoder decoder cells
            decoder_type: types for decoder cells
            num_decoder_symbols: the number of decoder symbols (this is used to deal with projection in feeding)
            encoder_masks: the masks that mask out encoder padding symbols, 
                           if it is None, the network is a simple static pointer network, otherwise dynamically
            feed_prev: a boolean tensor determining whether to feed forward in the decoding process
            loop_function: the feed_forward function
            multi_encoders: whether multiple encoders will be used, and how to combine information from them
            encoder_merge_method: 
                    0: merge as a full sequence encoding
                    1: encoding as multiple sequences, the second is initialized by the first bw state
                No matter which state is chosen, the pointer net output is always a sequence
        Returns:
            outs from the pointer net, types corresponding to decoder_types
    """

    with variable_scope.variable_scope(scope or "pointer_network") as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        if multi_encoders is None:
            enc_outs, enc_fw_state, enc_bw_state = bidirectional_static_rnn(fw_enc_cell, bw_enc_cell, enc_embeds, dtype=dtype)
            all_states = [enc_fw_state, enc_bw_state]

        elif encoder_merge_method == "parallel":
            enc_outs = [None] * len(enc_embeds)
            all_states = []

            for l in multi_encoders["segments"]:
                if all_states:
                    last_fw_state = all_states[-2]
                    last_bw_state = all_states[-1]
                    new_init_state = last_bw_state
                else:
                    new_init_state = None

                enc_embeds_seg = [enc_embeds[i] for i in l]
                encoder_masks_seg = [encoder_masks[i] for i in l]

                enc_outs_seg, enc_fw_state_seg, enc_bw_state_seg = bidirectional_static_rnn(fw_enc_cell, bw_enc_cell, enc_embeds_seg,
                                                                                             init_fw_state=new_init_state, dtype=dtype)

                for i in range(int(enc_outs_seg.get_shape()[0])):
                    enc_outs[l[i]] = enc_outs_seg[i]

                all_states.append(enc_fw_state_seg)
                all_states.append(enc_bw_state_seg)

        elif encoder_merge_method == "sequential":
            combined_l = []
            for l in multi_encoders["segments"]:
                combined_l.extend(l)

            enc_embeds_seg = [enc_embeds[i] for i in combined_l]
            encoder_masks_seg = [encoder_masks[i] for i in combined_l]

            enc_outs, enc_fw_states_seg, enc_bw_states_seg = bidirectional_static_rnn(fw_enc_cell, bw_enc_cell, enc_embeds_seg,  encoder_masks_seg, keep_all_states=True, dtype=dtype)

            all_states = []
            starting_index = 0

            for l in multi_encoders["segments"]:
                all_states.append(enc_fw_states_seg[starting_index + len(l) - 1])
                all_states.append(enc_bw_states_seg[starting_index + len(l) - 1])
                starting_index = starting_index + len(l)
        else:
            print("[Error] Encoder merge method is not speicified.")
            sys.exit(-1)

        state_size = ( fw_enc_cell.output_size + bw_enc_cell.output_size ) * ( 1 if multi_encoders is None else len(multi_encoders["segments"]))
        if weights is not None:
            w_combine = weights['encoder_w_combine']
            b_combine = weights['encoder_b_combine']
        else:
            w_combine = tf.get_variable("w_combine", initializer=tf.random_normal([state_size, dec_cell.output_size]))
            b_combine = tf.get_variable("b_combine", initializer=tf.random_normal([dec_cell.output_size]))

        combined_state = merge_multirnn_lstm_state(all_states, w_combine, b_combine)

        # prepare the encoder outs for attention.
        attention_states = tf.transpose(tf.stack(enc_outs), perm=[1,0,2])

        def _decoder_func(feed_flag, reuse_flag):
            # whether to reuse the decoder is determined by the flags
            with tf.variable_scope("attn_decoder", reuse=reuse_flag) as scope:
                return pointer_network_decoder(dec_cell, decoder_type, dec_embeds, combined_state, attention_states, 
                                               num_decoder_symbols, loop_function=loop_function if feed_flag else None, dtype=dtype, scope=scope,
                                               weights=weights)

        # we use the backward state as the input to the decoder, since we want the decoder carry for information about the task
        dec_outs, dec_state = tf.cond(tf.logical_not(feed_prev), 
                                      lambda: _decoder_func(False, False), 
                                      lambda: _decoder_func(True, True))

    return dec_outs
